use std::collections::HashMap;
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use incr_core::ErasedIncr;

use crate::Kind;

/// The xxh3-128 of `bytes`: every fingerprint the memo compares.
pub(crate) fn fp_of(bytes: &[u8]) -> u128 {
    xxhash_rust::xxh3::xxh3_128(bytes)
}

/// What a key's runtime node holds, as far as a memo can tell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DepFp {
    Known(u128),
    /// A mapped write of the key panicked part-way, so nothing that depends
    /// on it is trusted again in this process.
    Untrusted,
}

#[derive(Debug)]
pub(crate) struct DepEntry {
    pub(crate) handle: ErasedIncr,
    pub(crate) kind: Kind,
    pub(crate) fp: DepFp,
    /// An input's mapped stamp: its `changed_at` after the last mapped write
    /// that wrote. `None` for a collection and for an untrusted key.
    pub(crate) stamp: Option<u64>,
}

/// Every declared key of a store, with its node and fingerprint, while a
/// memo is open over the store.
#[derive(Default)]
pub(crate) struct DepIndex {
    map: RwLock<DepMap>,
}

#[derive(Default)]
pub(crate) struct DepMap {
    by_key: HashMap<String, DepEntry>,
    by_handle: HashMap<ErasedIncr, String>,
}

impl DepIndex {
    // Poisoning is ignored: a mapped write that panics marks the one key it
    // was writing untrusted before the hold is released.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn read(&self) -> RwLockReadGuard<'_, DepMap> {
        self.map.read().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, DepMap> {
        self.map.write().unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn fp(&self, key: &str) -> Option<DepFp> {
        self.read().entry(key).map(|entry| entry.fp)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn stamp(&self, key: &str) -> Option<u64> {
        self.read().entry(key).and_then(|entry| entry.stamp)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn key_of(&self, handle: ErasedIncr) -> Option<String> {
        self.read().by_handle.get(&handle).cloned()
    }

    #[cfg(test)]
    pub(crate) fn is_held(&self) -> bool {
        self.map.try_read().is_err()
    }
}

impl DepMap {
    pub(crate) fn entry(&self, key: &str) -> Option<&DepEntry> {
        self.by_key.get(key)
    }

    pub(crate) fn register(
        &mut self,
        key: &str,
        handle: ErasedIncr,
        kind: Kind,
        fp: DepFp,
        stamp: Option<u64>,
    ) {
        let stamp = stamp.filter(|_| fp != DepFp::Untrusted);
        let entry = DepEntry {
            handle,
            kind,
            fp,
            stamp,
        };
        if let Some(old) = self.by_key.insert(key.to_owned(), entry) {
            self.by_handle.remove(&old.handle);
        }
        self.by_handle.insert(handle, key.to_owned());
    }

    pub(crate) fn unregister(&mut self, key: &str) -> Option<DepEntry> {
        let entry = self.by_key.remove(key)?;
        self.by_handle.remove(&entry.handle);
        Some(entry)
    }

    /// Sets the fingerprint of a registered key; an untrusted one stays so.
    pub(crate) fn set_fp(&mut self, key: &str, fp: u128) {
        if let Some(entry) = self.by_key.get_mut(key) {
            if entry.fp != DepFp::Untrusted {
                entry.fp = DepFp::Known(fp);
            }
        }
    }

    /// Sets a registered input's fingerprint and mapped stamp after a write
    /// that wrote; an untrusted one stays so.
    pub(crate) fn set_written(&mut self, key: &str, fp: u128, stamp: u64) {
        if let Some(entry) = self.by_key.get_mut(key) {
            if entry.fp != DepFp::Untrusted {
                entry.fp = DepFp::Known(fp);
                entry.stamp = Some(stamp);
            }
        }
    }

    pub(crate) fn store_fp(&mut self, key: &str, fp: DepFp) {
        match fp {
            DepFp::Known(fp) => self.set_fp(key, fp),
            DepFp::Untrusted => self.untrust(key),
        }
    }

    pub(crate) fn untrust(&mut self, key: &str) {
        if let Some(entry) = self.by_key.get_mut(key) {
            entry.fp = DepFp::Untrusted;
            entry.stamp = None;
        }
    }
}

/// Marks `key` untrusted if dropped while unwinding, so a panic in user
/// code inside a mapped write never leaves a fingerprint that disagrees
/// with the runtime.
pub(crate) struct PanicGuard<'a> {
    deps: Option<&'a mut DepMap>,
    key: &'a str,
}

impl<'a> PanicGuard<'a> {
    pub(crate) fn new(deps: Option<&'a mut DepMap>, key: &'a str) -> Self {
        PanicGuard { deps, key }
    }

    pub(crate) fn deps(&mut self) -> Option<&mut DepMap> {
        self.deps.as_deref_mut()
    }

    /// Disarms the guard once the user code has returned.
    pub(crate) fn disarm(mut self) {
        self.deps = None;
    }
}

impl Drop for PanicGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            if let Some(deps) = self.deps.as_deref_mut() {
                deps.untrust(self.key);
            }
        }
    }
}
