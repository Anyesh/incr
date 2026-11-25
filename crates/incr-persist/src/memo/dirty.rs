//! Which live entries a checkpoint still has to save.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use incr_core::lock_order::{LockKind, Ordered};
use incr_core::ErasedIncr;

use super::table_core::{Encoded, KeyPart};

pub(crate) const SHARDS: usize = 16;

/// What [`Dirty::carry`] did with an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Carried {
    /// Not carried: computed again since the attempt, or not computed
    /// since the attempt before.
    No,
    /// Carried and counted unsaved again.
    Counted,
    /// Carried by an earlier attempt and not computed since.
    Before,
}

/// A dirty entry as an attempt copies it.
#[derive(Clone)]
pub(crate) struct Record {
    pub(crate) table: u32,
    pub(crate) handle: ErasedIncr,
    pub(crate) mark: u64,
    pub(crate) key: KeyPart,
}

struct Marked {
    table: u32,
    mark: u64,
    key: KeyPart,
}

#[derive(Default)]
struct Shard {
    /// Per entry, its table, key and the mark of its latest compute, in
    /// the order [`Dirty::take`] walks them.
    marks: Vec<(ErasedIncr, Marked)>,
    /// Each entry's index in `marks`.
    at: HashMap<ErasedIncr, usize>,
    /// How far the current attempt's [`Dirty::take`] has walked `marks`.
    /// A removal keeps the walked entries before it, so no entry the walk
    /// has yet to reach moves behind it.
    walked: usize,
    /// `marks`' length when the attempt began: entries added since belong
    /// to the next one, and a removal moves an unwalked entry only to a
    /// lower index, so the walk ends here or at the end, if sooner.
    end: usize,
    /// Entries [`Dirty::carry`] counted again, until they compute again,
    /// save or die, or the attempt fails; a subset of `marks`.
    carried: HashSet<ErasedIncr>,
}

impl Shard {
    fn get(&self, handle: &ErasedIncr) -> Option<&Marked> {
        self.at.get(handle).map(|&i| &self.marks[i].1)
    }

    fn get_mut(&mut self, handle: &ErasedIncr) -> Option<&mut Marked> {
        self.at.get(handle).map(|&i| &mut self.marks[i].1)
    }

    fn insert(&mut self, handle: ErasedIncr, marked: Marked) {
        self.at.insert(handle, self.marks.len());
        self.marks.push((handle, marked));
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.marks.swap(a, b);
        self.at.insert(self.marks[a].0, a);
        self.at.insert(self.marks[b].0, b);
    }

    fn remove(&mut self, handle: &ErasedIncr) -> Option<Marked> {
        let i = *self.at.get(handle)?;
        let last = self.marks.len() - 1;
        if i < self.walked {
            let edge = self.walked - 1;
            self.swap(i, edge);
            self.swap(edge, last);
            self.walked = edge;
        } else {
            self.swap(i, last);
        }
        self.at.remove(handle);
        self.carried.remove(handle);
        self.marks.pop().map(|(_, marked)| marked)
    }

    fn values(&self) -> impl Iterator<Item = &Marked> {
        self.marks.iter().map(|(_, marked)| marked)
    }
}

/// The live entries computed since a checkpoint last saved them, each
/// with the mark of its latest compute, sharded by handle slot so that
/// computes on different threads rarely meet. Marks rise by one per
/// compute; a checkpoint attempt notes the mark it started at, so an entry
/// whose mark is above it was computed since that attempt.
///
/// The counters change only under a shard lock, and `attempt` and
/// `previous` only under every shard lock, so a reader holding any shard
/// lock sees them consistent. The shard locks are leaves of the lock
/// order: nothing is taken under them.
pub(crate) struct Dirty {
    marks: AtomicU64,
    attempt: AtomicU64,
    previous: AtomicU64,
    /// Equals the number of entries whose mark is above `attempt`.
    unsaved: AtomicU64,
    /// `previous` and `attempt` before the latest attempt, for
    /// [`Self::failed`]; written under every shard lock.
    undo: (AtomicU64, AtomicU64),
    shards: Box<[Mutex<Shard>]>,
}

impl Dirty {
    pub(crate) fn new() -> Self {
        Dirty {
            marks: AtomicU64::new(0),
            attempt: AtomicU64::new(0),
            previous: AtomicU64::new(0),
            unsaved: AtomicU64::new(0),
            undo: (AtomicU64::new(0), AtomicU64::new(0)),
            shards: (0..SHARDS).map(|_| Mutex::default()).collect(),
        }
    }

    fn shard(&self, handle: ErasedIncr) -> Ordered<MutexGuard<'_, Shard>> {
        let shard = &self.shards[handle.slot() as usize % SHARDS];
        Ordered::take(LockKind::MemoDirty, || {
            shard.lock().unwrap_or_else(PoisonError::into_inner)
        })
    }

    fn all(&self) -> Ordered<Vec<MutexGuard<'_, Shard>>> {
        Ordered::take(LockKind::MemoDirty, || {
            self.shards
                .iter()
                .map(|shard| shard.lock().unwrap_or_else(PoisonError::into_inner))
                .collect()
        })
    }

    /// Records a compute of `handle`; returns the unsaved count after it.
    pub(crate) fn computed(&self, table: u32, handle: ErasedIncr, key: &KeyPart) -> u64 {
        let mut shard = self.shard(handle);
        let mark = self.marks.fetch_add(1, Relaxed) + 1;
        shard.carried.remove(&handle);
        let previous = match shard.get_mut(&handle) {
            Some(marked) => Some(std::mem::replace(&mut marked.mark, mark)),
            None => {
                let key = Arc::clone(key);
                shard.insert(handle, Marked { table, mark, key });
                None
            }
        };
        if previous.is_none_or(|mark| mark <= self.attempt.load(Relaxed)) {
            self.unsaved.fetch_add(1, Relaxed) + 1
        } else {
            self.unsaved.load(Relaxed)
        }
    }

    /// Marks a clean entry the backend lost dirty and counts it as unsaved,
    /// as a compute would; returns the unsaved count after it.
    pub(crate) fn lost(&self, table: u32, handle: ErasedIncr, key: Vec<u8>) -> u64 {
        let key: KeyPart = Arc::new(Encoded(key));
        self.computed(table, handle, &key)
    }

    /// Keeps `handle` dirty without counting it as unsaved: an entry a
    /// checkpoint evicted, which may be evicted again at every checkpoint.
    pub(crate) fn evicted(&self, table: u32, handle: ErasedIncr, key: Vec<u8>) {
        let mut shard = self.shard(handle);
        if shard.get(&handle).is_none() {
            let key = Arc::new(Encoded(key));
            shard.insert(
                handle,
                Marked {
                    table,
                    mark: 0,
                    key,
                },
            );
        }
    }

    /// Starts a checkpoint attempt: removes entries `alive` refuses and
    /// returns a copy of the rest, by table and then mark, so a checkpoint
    /// writes a table's entries together and in a repeatable order. The
    /// removed entries' keys go to `dead`: a dead node's may be the last
    /// reference to a user key, so the caller drops them outside its locks.
    pub(crate) fn attempt(
        &self,
        alive: impl Fn(ErasedIncr) -> bool,
        dead: &mut Vec<KeyPart>,
    ) -> Vec<Record> {
        let mut shards = self.all();
        self.start(&mut shards);
        let attempt = self.attempt.load(Relaxed);
        let mut out = Vec::new();
        for shard in shards.iter_mut() {
            walk(shard, attempt, usize::MAX, &alive, dead, &mut out);
        }
        out.sort_unstable_by_key(|record| (record.table, record.mark));
        out
    }

    /// Starts a checkpoint attempt without copying it: the entries whose
    /// mark is at or below the attempt's are its, and [`Self::take`]
    /// copies them a piece of a shard at a time.
    pub(crate) fn begin(&self) {
        let mut shards = self.all();
        self.start(&mut shards);
    }

    /// [`Self::attempt`] for the next piece of shard `shard`, below
    /// [`SHARDS`]: walks on over at most `most` of its entries from where
    /// the attempt's last take of it stopped. Returns the piece by table
    /// and then mark, and whether the walk reached the shard's end. Entries
    /// computed since the attempt began are left for the next one.
    pub(crate) fn take(
        &self,
        shard: usize,
        most: usize,
        alive: impl Fn(ErasedIncr) -> bool,
        dead: &mut Vec<KeyPart>,
    ) -> (Vec<Record>, bool) {
        let mut guard = Ordered::take(LockKind::MemoDirty, || {
            self.shards[shard]
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
        });
        let attempt = self.attempt.load(Relaxed);
        let mut out = Vec::new();
        let done = walk(&mut guard, attempt, most, &alive, dead, &mut out);
        out.sort_unstable_by_key(|record| (record.table, record.mark));
        (out, done)
    }

    /// Moves the counters to a new attempt and restarts every shard's walk;
    /// the caller holds every shard lock.
    fn start(&self, shards: &mut [MutexGuard<'_, Shard>]) {
        for shard in shards.iter_mut() {
            shard.walked = 0;
            shard.end = shard.marks.len();
        }
        let (previous, attempt) = (self.previous.load(Relaxed), self.attempt.load(Relaxed));
        self.undo.0.store(previous, Relaxed);
        self.undo.1.store(attempt, Relaxed);
        self.previous.store(attempt, Relaxed);
        self.attempt.store(self.marks.load(Relaxed), Relaxed);
        self.unsaved.store(0, Relaxed);
    }

    /// Takes back the latest attempt, which saved nothing: every entry
    /// computed since the attempt before it counts as unsaved again, and
    /// the next attempt may carry what this one carried.
    pub(crate) fn failed(&self) {
        let mut shards = self.all();
        for shard in shards.iter_mut() {
            shard.carried.clear();
        }
        let attempt = self.undo.1.load(Relaxed);
        self.previous.store(self.undo.0.load(Relaxed), Relaxed);
        self.attempt.store(attempt, Relaxed);
        let unsaved = shards
            .iter()
            .flat_map(|shard| shard.values())
            .filter(|marked| marked.mark > attempt)
            .count();
        self.unsaved.store(unsaved as u64, Relaxed);
    }

    /// Counts again an entry the attempt found dirty at `mark` but could
    /// not inspect, if it computed since the attempt before: its compute
    /// may not have landed yet. Once per compute, so an entry left stale
    /// is not counted at every attempt.
    pub(crate) fn carry(&self, record: &Record) -> Carried {
        let mut shard = self.shard(record.handle);
        if shard
            .get(&record.handle)
            .is_none_or(|marked| marked.mark != record.mark)
        {
            return Carried::No;
        }
        if shard.carried.contains(&record.handle) {
            return Carried::Before;
        }
        if record.mark <= self.previous.load(Relaxed) {
            return Carried::No;
        }
        shard.carried.insert(record.handle);
        let current = shard.get_mut(&record.handle).expect("checked present");
        current.mark = self.marks.fetch_add(1, Relaxed) + 1;
        self.unsaved.fetch_add(1, Relaxed);
        Carried::Counted
    }

    /// Counts as unsaved a dirty entry that landed with no compute, unless
    /// something counted it since the attempt began: an attempt that
    /// carried it again did not, and its landing is what lets a
    /// checkpoint save it.
    pub(crate) fn landed(&self, handle: ErasedIncr) {
        let mut shard = self.shard(handle);
        let attempt = self.attempt.load(Relaxed);
        if let Some(current) = shard
            .get_mut(&handle)
            .filter(|marked| marked.mark <= attempt)
        {
            current.mark = self.marks.fetch_add(1, Relaxed) + 1;
            self.unsaved.fetch_add(1, Relaxed);
        }
    }

    /// `handle` reached the backend as of `mark`; a later compute keeps it.
    /// Returns the removed entry's key, which may be the last reference to
    /// a user key, for the caller to drop outside its locks.
    #[must_use]
    pub(crate) fn saved(&self, handle: ErasedIncr, mark: u64) -> Option<KeyPart> {
        let mut shard = self.shard(handle);
        if shard.get(&handle).is_some_and(|marked| marked.mark == mark) {
            shard.remove(&handle).map(|marked| marked.key)
        } else {
            None
        }
    }

    pub(crate) fn unsaved(&self) -> u64 {
        self.unsaved.load(Relaxed)
    }

    /// The entries whose mark is above the attempt's, which `unsaved`
    /// equals.
    #[cfg(test)]
    pub(crate) fn above_attempt(&self) -> u64 {
        let attempt = self.attempt.load(Relaxed);
        self.all()
            .iter()
            .flat_map(|shard| shard.values())
            .filter(|marked| marked.mark > attempt)
            .count() as u64
    }
}

/// Walks `shard` on from where its walk stopped, over at most `most`
/// entries: removes those at or below `attempt` that `alive` refuses,
/// their keys to `dead`, and copies the rest of them to `out`. Returns
/// whether the walk reached the end.
fn walk(
    shard: &mut Shard,
    attempt: u64,
    most: usize,
    alive: &impl Fn(ErasedIncr) -> bool,
    dead: &mut Vec<KeyPart>,
    out: &mut Vec<Record>,
) -> bool {
    for _ in 0..most {
        if shard.walked >= shard.end {
            return true;
        }
        let Some((handle, marked)) = shard.marks.get(shard.walked) else {
            return true;
        };
        let handle = *handle;
        if marked.mark <= attempt && !alive(handle) {
            dead.extend(shard.remove(&handle).map(|marked| marked.key));
            continue;
        }
        if marked.mark <= attempt {
            out.push(Record {
                table: marked.table,
                handle,
                mark: marked.mark,
                key: Arc::clone(&marked.key),
            });
        }
        shard.walked += 1;
    }
    shard.walked >= shard.end.min(shard.marks.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use incr_core::{Local, Runtime};

    fn handles(n: u64) -> (Runtime<Local>, Vec<ErasedIncr>) {
        let rt = Runtime::<Local>::new();
        let handles = (0..n).map(|i| rt.create_input(i).erase()).collect();
        (rt, handles)
    }

    fn mark(records: &[Record], handle: ErasedIncr) -> Record {
        records.iter().find(|r| r.handle == handle).unwrap().clone()
    }

    fn key() -> KeyPart {
        Arc::new(Encoded(Vec::new()))
    }

    #[test]
    fn unsaved_counts_entries_computed_since_the_attempt() {
        let (_rt, h) = handles(3);
        let dirty = Dirty::new();
        assert_eq!(dirty.computed(0, h[0], &key()), 1);
        assert_eq!(dirty.computed(0, h[0], &key()), 1);
        assert_eq!(dirty.computed(0, h[1], &key()), 2);
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(dirty.unsaved(), 0);
        assert_eq!(dirty.computed(0, h[0], &key()), 1);
        dirty.evicted(0, h[2], Vec::new());
        assert_eq!(dirty.unsaved(), 1);
        drop(dirty.saved(h[0], mark(&marks, h[0]).mark));
        drop(dirty.saved(h[1], mark(&marks, h[1]).mark));
        let left = dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(left.len(), 2);
        assert!(left.iter().any(|r| r.handle == h[0]) && left.iter().any(|r| r.handle == h[2]));
    }

    #[test]
    fn a_lost_entry_counts_once_until_the_next_attempt() {
        let (_rt, h) = handles(2);
        let dirty = Dirty::new();
        dirty.computed(0, h[0], &key());
        dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(dirty.lost(0, h[1], Vec::new()), 1);
        assert_eq!(dirty.lost(0, h[1], Vec::new()), 1);
        assert_eq!(dirty.lost(0, h[0], Vec::new()), 2);
        assert_eq!(dirty.attempt(|_| true, &mut Vec::new()).len(), 2);
    }

    #[test]
    fn a_failed_attempt_counts_its_entries_again_once() {
        let (_rt, h) = handles(3);
        let dirty = Dirty::new();
        dirty.computed(0, h[0], &key());
        dirty.computed(0, h[1], &key());
        dirty.attempt(|_| true, &mut Vec::new());
        dirty.computed(0, h[1], &key());
        dirty.computed(0, h[2], &key());
        dirty.failed();
        assert_eq!(dirty.unsaved(), 3);
        assert_eq!(dirty.computed(0, h[0], &key()), 3);
    }

    #[test]
    fn an_entry_not_yet_landed_is_carried_once_per_compute() {
        let (_rt, h) = handles(2);
        let dirty = Dirty::new();
        dirty.computed(0, h[0], &key());
        dirty.evicted(0, h[1], Vec::new());
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(dirty.carry(&mark(&marks, h[0])), Carried::Counted);
        assert_eq!(dirty.carry(&mark(&marks, h[1])), Carried::No);
        assert_eq!(dirty.unsaved(), 1);
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(dirty.carry(&mark(&marks, h[0])), Carried::Before);
        assert_eq!(dirty.unsaved(), 0);
        dirty.computed(0, h[0], &key());
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(dirty.carry(&mark(&marks, h[0])), Carried::Counted);
        assert_eq!(dirty.unsaved(), 1);
    }

    #[test]
    fn an_entry_carried_twice_counts_once_more_when_it_lands() {
        let (_rt, h) = handles(2);
        let dirty = Dirty::new();
        dirty.computed(0, h[0], &key());
        dirty.computed(0, h[1], &key());
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        dirty.carry(&mark(&marks, h[0]));
        dirty.carry(&mark(&marks, h[1]));
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(dirty.carry(&mark(&marks, h[0])), Carried::Before);
        assert_eq!(dirty.carry(&mark(&marks, h[1])), Carried::Before);
        dirty.computed(0, h[1], &key());
        assert_eq!(dirty.unsaved(), 1);
        dirty.landed(h[0]);
        dirty.landed(h[1]);
        assert_eq!(dirty.unsaved(), 2);
        dirty.failed();
        assert_eq!(dirty.unsaved(), 2);
    }

    #[test]
    fn a_landing_counts_an_entry_once_per_attempt() {
        let (_rt, h) = handles(1);
        let dirty = Dirty::new();
        dirty.computed(0, h[0], &key());
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        dirty.carry(&mark(&marks, h[0]));
        dirty.landed(h[0]);
        assert_eq!(dirty.unsaved(), 1);
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(dirty.carry(&mark(&marks, h[0])), Carried::Before);
        dirty.landed(h[0]);
        dirty.landed(h[0]);
        assert_eq!(dirty.unsaved(), 1);
    }

    #[test]
    fn a_failed_attempt_lets_its_carried_entries_be_carried_again() {
        let (_rt, h) = handles(1);
        let dirty = Dirty::new();
        dirty.computed(0, h[0], &key());
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(dirty.carry(&mark(&marks, h[0])), Carried::Counted);
        dirty.failed();
        assert_eq!(dirty.unsaved(), 1);
        let marks = dirty.attempt(|_| true, &mut Vec::new());
        assert_eq!(dirty.carry(&mark(&marks, h[0])), Carried::Counted);
        assert_eq!(dirty.unsaved(), 1);
    }
    #[test]
    fn a_take_walks_a_shard_in_pieces_and_loses_no_entry_to_removals() {
        let rt = Runtime::<Local>::new();
        let h: Vec<ErasedIncr> = (0..4_000_u64)
            .map(|i| rt.create_input(i).erase())
            .filter(|h| h.slot() as usize % SHARDS == 3)
            .take(120)
            .collect();
        let dirty = Dirty::new();
        for &handle in &h[..100] {
            dirty.computed(0, handle, &key());
        }
        dirty.begin();
        let mut taken: Vec<Record> = Vec::new();
        let mut gone = Vec::new();
        let mut pieces = 0;
        loop {
            let (piece, done) = dirty.take(3, 10, |_| true, &mut Vec::new());
            assert!(piece.len() <= 10);
            taken.extend(piece);
            pieces += 1;
            let behind = &taken[pieces - 1];
            assert!(dirty.saved(behind.handle, behind.mark).is_some());
            let ahead = h[100 - pieces];
            if !taken.iter().any(|r| r.handle == ahead) {
                assert!(dirty.saved(ahead, (101 - pieces) as u64).is_some());
                gone.push(ahead);
            }
            dirty.computed(0, h[100 + pieces % 20], &key());
            if done {
                break;
            }
        }
        assert!(pieces >= 6, "{pieces} pieces");
        let mut handles: Vec<ErasedIncr> = taken.iter().map(|r| r.handle).collect();
        handles.sort_unstable_by_key(|h| h.slot());
        handles.dedup();
        assert_eq!(handles.len(), taken.len(), "an entry was taken twice");
        let mut want: Vec<ErasedIncr> = h[..100]
            .iter()
            .copied()
            .filter(|h| !gone.contains(h))
            .collect();
        want.sort_unstable_by_key(|h| h.slot());
        assert_eq!(handles, want);
    }

    #[test]
    fn a_take_finishes_the_shard_however_many_entries_land_behind_it() {
        let rt = Runtime::<Local>::new();
        let h: Vec<ErasedIncr> = (0..20_000_u64)
            .map(|i| rt.create_input(i).erase())
            .filter(|h| h.slot() as usize % SHARDS == 5)
            .take(1_000)
            .collect();
        let dirty = Dirty::new();
        for &handle in &h[..100] {
            dirty.computed(0, handle, &key());
        }
        dirty.begin();
        let mut taken = 0;
        let mut next = 100;
        for _ in 0..20 {
            let (piece, done) = dirty.take(5, 10, |_| true, &mut Vec::new());
            taken += piece.len();
            if done {
                assert_eq!(taken, 100);
                return;
            }
            for &handle in &h[next..next + 20] {
                dirty.computed(0, handle, &key());
            }
            next += 20;
        }
        panic!("the walk took {taken} of 100 and never reached the attempt's end");
    }
}
