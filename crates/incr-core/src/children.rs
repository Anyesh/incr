//! A creator's index of its owned children: from a child's identity
//! (key, key type, value type, kind) to its node, with the run that
//! created it and the last run that touched it.
//!
//! Only the thread holding the creator's compute claim touches an index,
//! through a [`Cells::DataCell`]. A claim is taken by an Acquire CAS
//! that reads from the previous holder's Release landing, so every
//! access happens-after the previous holder's last one.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use crate::cells::Cells;
use crate::fx::FxBuildHasher;
use crate::handle::ErasedIncr;
use crate::table::DataCell;

pub(crate) struct ChildIndex<C: Cells> {
    state: C::DataCell<IndexState>,
}

impl<C: Cells> ChildIndex<C> {
    /// An index created inside the run it is created for, which is its
    /// first.
    pub(crate) fn new() -> Self {
        Self {
            state: C::DataCell::new(IndexState {
                epoch: 1,
                landed: 0,
                maps: Vec::new(),
                orphans: Vec::new(),
            }),
        }
    }

    /// Runs `f` on the index. The caller holds the creator's claim, and
    /// `f` must not reach the index again.
    #[inline]
    pub(crate) fn with<R>(&self, f: impl FnOnce(&mut IndexState) -> R) -> R {
        // SAFETY: only the creator's claim holder reaches the index (see
        // the module docs), and `f` does not reenter it.
        self.state.with_mut(|s| f(unsafe { &mut *s }))
    }
}

/// The identity marker of one map: a child's key type, value type and
/// kind.
struct Slot<K, T, const QUERY: bool>(PhantomData<fn() -> (K, T)>);

#[derive(Copy, Clone)]
struct Entry {
    handle: ErasedIncr,
    created: u64,
    touched: u64,
}

pub(crate) struct IndexState {
    /// The current run's number; every run of the creator bumps it.
    pub(crate) epoch: u64,
    /// The epoch at the creator's last landing. Entries created after it
    /// belong to the claim in progress.
    landed: u64,
    maps: Vec<(TypeId, Box<dyn ChildMap>)>,
    /// Owned children no key names any more, because the attempt that
    /// created them was discarded while something still read them.
    orphans: Vec<ErasedIncr>,
}

impl IndexState {
    /// The child bound to `key`, marked touched by this run.
    pub(crate) fn find<K, T, const QUERY: bool>(&mut self, key: &K) -> Option<ErasedIncr>
    where
        K: Hash + Eq + Send + Sync + 'static,
        T: 'static,
    {
        let epoch = self.epoch;
        let map = self.map::<K, T, QUERY>()?;
        let entry = map.get_mut(key)?;
        entry.touched = epoch;
        Some(entry.handle)
    }

    /// Binds `key` to `handle`, created and touched by this run. A
    /// binding it displaces (made by a reentrant create of the same key)
    /// becomes an orphan.
    pub(crate) fn insert<K, T, const QUERY: bool>(&mut self, key: K, handle: ErasedIncr)
    where
        K: Hash + Eq + Send + Sync + 'static,
        T: 'static,
    {
        let id = TypeId::of::<Slot<K, T, QUERY>>();
        let at = match self.maps.iter().position(|(t, _)| *t == id) {
            Some(at) => at,
            None => {
                self.maps
                    .push((id, Box::new(KeyMap::<K>(HashMap::default()))));
                self.maps.len() - 1
            }
        };
        let epoch = self.epoch;
        let map = downcast::<K>(&mut self.maps[at].1);
        let entry = Entry {
            handle,
            created: epoch,
            touched: epoch,
        };
        if let Some(old) = map.insert(key, entry) {
            self.orphans.push(old.handle);
        }
    }

    /// Unbinds `key` from a child the caller found deleted.
    pub(crate) fn unbind<K, T, const QUERY: bool>(&mut self, key: &K) -> Option<K>
    where
        K: Hash + Eq + Send + Sync + 'static,
        T: 'static,
    {
        self.map::<K, T, QUERY>()?.remove_entry(key).map(|(k, _)| k)
    }

    fn map<K, T, const QUERY: bool>(&mut self) -> Option<&mut HashMap<K, Entry, FxBuildHasher>>
    where
        K: Hash + Eq + Send + Sync + 'static,
        T: 'static,
    {
        let id = TypeId::of::<Slot<K, T, QUERY>>();
        let (_, map) = self.maps.iter_mut().find(|(t, _)| *t == id)?;
        Some(downcast::<K>(map))
    }

    /// The children a Clean landing may delete: every bound child this
    /// run did not touch, and every orphan.
    pub(crate) fn sweep_candidates(&self) -> Vec<ErasedIncr> {
        let epoch = self.epoch;
        let mut out = self.orphans.clone();
        for (_, map) in &self.maps {
            map.select(&mut |e| e.touched != epoch, &mut out);
        }
        out
    }

    /// Unbinds every child the claim in progress created, making each an
    /// orphan, and returns the removed keys for the caller's bag.
    pub(crate) fn unbind_created(&mut self) -> Vec<Box<dyn Any>> {
        let landed = self.landed;
        let mut keys = Vec::new();
        for (_, map) in &mut self.maps {
            if let Some(removed) = map.remove(&mut |e| e.created > landed, &mut self.orphans) {
                keys.push(removed);
            }
        }
        keys
    }

    /// Drops the bindings and orphans whose child `gone` reports deleted,
    /// returning the removed keys for the caller's bag.
    pub(crate) fn forget(&mut self, gone: &dyn Fn(ErasedIncr) -> bool) -> Vec<Box<dyn Any>> {
        self.orphans.retain(|&h| !gone(h));
        let mut keys = Vec::new();
        let mut unbound = Vec::new();
        for (_, map) in &mut self.maps {
            if let Some(removed) = map.remove(&mut |e| gone(e.handle), &mut unbound) {
                keys.push(removed);
            }
        }
        keys
    }

    /// Ends the claim in progress: its children are now the creator's.
    pub(crate) fn land(&mut self) {
        self.landed = self.epoch;
    }

    pub(crate) fn orphans(&self) -> &[ErasedIncr] {
        &self.orphans
    }

    /// Every child the index names, bound or orphaned.
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn children(&self) -> Vec<ErasedIncr> {
        let mut out = self.orphans.clone();
        for (_, map) in &self.maps {
            map.select(&mut |_| true, &mut out);
        }
        out
    }
}

fn downcast<K: Hash + Eq + Send + Sync + 'static>(
    map: &mut Box<dyn ChildMap>,
) -> &mut HashMap<K, Entry, FxBuildHasher> {
    &mut map
        .as_any_mut()
        .downcast_mut::<KeyMap<K>>()
        .expect("incr-core: a child map's type id names its key type")
        .0
}

/// One key type's bindings, erased so that the sweep can scan every map.
trait ChildMap: Send + Sync {
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn select(&self, pick: &mut dyn FnMut(&Entry) -> bool, out: &mut Vec<ErasedIncr>);

    /// Removes every entry `pick` chooses, appending its handle to
    /// `handles`, and returns the removed keys boxed, if any.
    fn remove(
        &mut self,
        pick: &mut dyn FnMut(&Entry) -> bool,
        handles: &mut Vec<ErasedIncr>,
    ) -> Option<Box<dyn Any>>;
}

struct KeyMap<K>(HashMap<K, Entry, FxBuildHasher>);

impl<K: Hash + Eq + Send + Sync + 'static> ChildMap for KeyMap<K> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn select(&self, pick: &mut dyn FnMut(&Entry) -> bool, out: &mut Vec<ErasedIncr>) {
        out.extend(self.0.values().filter(|e| pick(e)).map(|e| e.handle));
    }

    fn remove(
        &mut self,
        pick: &mut dyn FnMut(&Entry) -> bool,
        handles: &mut Vec<ErasedIncr>,
    ) -> Option<Box<dyn Any>> {
        let removed: Vec<K> = self
            .0
            .extract_if(|_, e| pick(e))
            .map(|(k, e)| {
                handles.push(e.handle);
                k
            })
            .collect();
        (!removed.is_empty()).then(|| Box::new(removed) as Box<dyn Any>)
    }
}
