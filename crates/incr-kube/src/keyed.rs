//! `KeyedCollection<K, V, Ver>`: Kubernetes-style keyed replace/delete
//! semantics layered over `IncrCollection`'s multiset.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, PoisonError, RwLock};

use incr_concurrent::{IncrCollection, Runtime};

use crate::entry::Entry;

/// Result of a `KeyedCollection::upsert` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsertOutcome {
    /// No prior entry for this key; the value was inserted.
    Inserted,
    /// A prior entry existed at an older version and was replaced.
    Replaced,
    /// `version` was not newer than the stored version; the event was
    /// dropped.
    Stale,
}

/// Wraps an `IncrCollection<Entry<K, V, Ver>>` with a `key -> Entry`
/// index, so `upsert`/`remove` can address `delete(old)` by key alone
/// instead of requiring the caller to hand back the exact prior value.
///
/// The index is the answer to "does this need a third index beyond
/// kube-rs's `Store` and incr's own collection": yes. It's a plain
/// `RwLock`, not lock-free, because there is exactly one writer thread
/// in this design (all writes serialize trivially) while `get()` reads
/// among themselves should not block each other. `get()` does still
/// block behind an in-progress writer: the write critical section spans
/// the index update plus the single `replace` call (index lock, then
/// the collection's own log lock, then its `rt.set` dirty walk), held
/// for that whole span rather than released between steps, because
/// releasing it earlier would let two `upsert` calls on the same key
/// interleave and leave two live entries.
///
/// **Relist reconciliation is the caller's job, via `retain`.** `remove`
/// drops the index entry outright, so no version floor survives a
/// delete: within one ordered watch stream a stale re-apply after a
/// real delete can't happen, but across a relist (kube-rs reconnecting
/// its watch and replaying from a snapshot) it plausibly could,
/// resurrecting a deleted object under a stale version. `retain` is the
/// primitive that closes this: a caller that owns relist-boundary
/// semantics (the reflector integration layer, not this generic engine
/// layer) can prune anything not observed in a fresh full listing.
/// `KeyedCollection` itself never calls `retain` on its own; nothing
/// here assumes a relist happened unless the caller drives one.
pub struct KeyedCollection<K, V, Ver = u64>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    Ver: Copy + Ord + Hash + Send + Sync + 'static,
{
    collection: IncrCollection<Entry<K, V, Ver>>,
    index: RwLock<HashMap<K, Entry<K, V, Ver>>>,
}

impl<K, V, Ver> KeyedCollection<K, V, Ver>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    Ver: Copy + Ord + Hash + Send + Sync + 'static,
{
    /// Create an empty keyed collection in `rt`.
    pub fn new(rt: &Runtime) -> Self {
        Self {
            collection: rt.create_collection(),
            index: RwLock::new(HashMap::new()),
        }
    }

    /// Apply a keyed replace event. Rejects `version <= stored.version`
    /// as stale rather than applying it.
    pub fn upsert(&self, rt: &Runtime, key: K, version: Ver, value: V) -> UpsertOutcome {
        let mut index = self.index.write().unwrap_or_else(PoisonError::into_inner);
        match index.get(&key) {
            Some(existing) if existing.version >= version => UpsertOutcome::Stale,
            Some(existing) => {
                let new_entry = Entry::new(key.clone(), version, value);
                let deleted = self
                    .collection
                    .replace(rt, Some(existing), new_entry.clone());
                debug_assert!(
                    deleted,
                    "incr-kube: index had an entry for this key that replace() didn't find \
                     in the collection; index and collection have drifted out of sync"
                );
                index.insert(key, new_entry);
                UpsertOutcome::Replaced
            }
            None => {
                let new_entry = Entry::new(key.clone(), version, value);
                self.collection.replace(rt, None, new_entry.clone());
                index.insert(key, new_entry);
                UpsertOutcome::Inserted
            }
        }
    }

    /// Apply a keyed delete event. Returns whether a live entry for
    /// `key` was found and removed.
    pub fn remove(&self, rt: &Runtime, key: &K) -> bool {
        let mut index = self.index.write().unwrap_or_else(PoisonError::into_inner);
        match index.remove(key) {
            Some(entry) => {
                self.collection.delete(rt, &entry);
                true
            }
            None => false,
        }
    }

    /// Direct point lookup, bypassing the incr graph: a synchronous
    /// read of the index, not an incrementally maintained value.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        let index = self.index.read().unwrap_or_else(PoisonError::into_inner);
        index.get(key).map(|entry| Arc::clone(&entry.value))
    }

    /// A read-only view of the underlying collection: `filter`, `map`,
    /// `join`, `group_by`, `aggregate`, `count`, and `reduce` all work
    /// on it. `insert`/`delete`/`replace` panic on it (it's an
    /// `as_view()` clone), so it can't be used to desync the index.
    pub fn collection(&self) -> IncrCollection<Entry<K, V, Ver>> {
        self.collection.as_view()
    }

    /// Highest version currently stored, or `None` if the collection is
    /// empty. Lets a caller reseed a loop-owned version counter (a
    /// `drive_reflector` restart against this same collection, say)
    /// above whatever's already indexed, instead of starting back at the
    /// type's zero value and having every event through the old
    /// high-water mark rejected as `Stale`.
    pub fn max_version(&self) -> Option<Ver> {
        self.index
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .map(|entry| entry.version)
            .max()
    }

    /// Number of live keys.
    pub fn len(&self) -> usize {
        self.index
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Whether there are no live keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove every currently-indexed key for which `keep` returns
    /// false. Meant for reconciling against an authoritative full
    /// listing (a Kubernetes relist): anything not in that listing gets
    /// pruned. Returns how many keys were removed.
    ///
    /// Each removal is a real, individual delete, not a batched
    /// operation: a reader mid-`retain` sees a true intermediate state
    /// (some keys already pruned, some not yet), not a transient
    /// artifact the way an unbatched same-key `replace` would expose.
    pub fn retain(&self, rt: &Runtime, keep: impl Fn(&K) -> bool) -> usize {
        let mut index = self.index.write().unwrap_or_else(PoisonError::into_inner);
        let stale: Vec<K> = index.keys().filter(|k| !keep(k)).cloned().collect();
        let mut removed = 0;
        for key in stale {
            if let Some(entry) = index.remove(&key) {
                self.collection.delete(rt, &entry);
                removed += 1;
            }
        }
        removed
    }
}
