//! `Entry<K, V, Ver>`: the multiset element `KeyedCollection` stores
//! inside an `IncrCollection`. Equality and hashing are defined on
//! `(key, version)` only; `value` is deliberately excluded from both, so
//! `V` never needs to implement `Hash`/`Eq` itself. Most Kubernetes
//! resource types don't. Two watch events for the same object at the
//! same version collapse to one multiset entry, matching Kubernetes'
//! own dedup semantics.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// One versioned value under a key.
///
/// `Ver` defaults to `u64`. `drive_reflector` (this crate's kube-rs
/// integration) never parses Kubernetes' `resourceVersion` string into
/// it: a single watch stream already delivers events for a given object
/// in order, so a loop-owned monotonic counter is a correct `Ver` on
/// its own. A caller driving `KeyedCollection` directly, outside
/// `drive_reflector`, is free to supply any `Ver` with its own ordering
/// guarantee instead.
pub struct Entry<K, V, Ver = u64> {
    pub(crate) key: K,
    pub(crate) version: Ver,
    pub(crate) value: Arc<V>,
}

impl<K, V, Ver> Entry<K, V, Ver> {
    pub(crate) fn new(key: K, version: Ver, value: V) -> Self {
        Self {
            key,
            version,
            value: Arc::new(value),
        }
    }

    /// The key this entry is stored under.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// The version this entry was last replaced at.
    pub fn version(&self) -> &Ver {
        &self.version
    }

    /// The value itself, shared via `Arc` rather than cloned.
    pub fn value(&self) -> &Arc<V> {
        &self.value
    }
}

impl<K: Clone, V, Ver: Clone> Clone for Entry<K, V, Ver> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            version: self.version.clone(),
            value: Arc::clone(&self.value),
        }
    }
}

impl<K: PartialEq, V, Ver: PartialEq> PartialEq for Entry<K, V, Ver> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.version == other.version
    }
}

impl<K: Eq, V, Ver: Eq> Eq for Entry<K, V, Ver> {}

impl<K: Hash, V, Ver: Hash> Hash for Entry<K, V, Ver> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
        self.version.hash(state);
    }
}

impl<K: std::fmt::Debug, V, Ver: std::fmt::Debug> std::fmt::Debug for Entry<K, V, Ver> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry")
            .field("key", &self.key)
            .field("version", &self.version)
            .finish()
    }
}
