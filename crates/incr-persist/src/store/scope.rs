use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::{Arc, PoisonError};

use incr_core::persist_hooks::changed_at_erased;
use incr_core::{Cells, Incr, IncrCollection, KeyedCollection, Pair, Value};

use super::{Input, KeyRef, PersistValue, PersistedCollection, PersistedKeyed, Store};
use crate::{Backend, Error, Kind, Result};

/// A window in which the caller changes the runtime directly, around the
/// log, and restores every change exactly before [`close`](Self::close).
///
/// While it is open the store is inert: every call that writes, reads the
/// log or opens the memo's index fails with [`Error::UnloggedScopeOpen`],
/// and a memo over the store computes lookups fresh instead of restoring,
/// and refuses to checkpoint. Dropping a scope without closing it poisons
/// the store, because the runtime may then hold values the log does not.
///
/// Call [`touched`](Self::touched) for an input before changing it, so that
/// `close` can tell the restored value from one changed earlier.
#[must_use = "dropping an unlogged scope without closing it poisons the store"]
pub struct UnloggedScope<'s, C: Cells, B: Backend> {
    store: StoreRef<'s, C, B>,
    touched: HashMap<u32, bool>,
    closed: bool,
}

pub(super) enum StoreRef<'s, C: Cells, B: Backend> {
    Borrowed(&'s Store<C, B>),
    Shared(Arc<Store<C, B>>),
}

impl<C: Cells, B: Backend> Deref for StoreRef<'_, C, B> {
    type Target = Store<C, B>;

    fn deref(&self) -> &Store<C, B> {
        match self {
            StoreRef::Borrowed(store) => store,
            StoreRef::Shared(store) => store,
        }
    }
}

impl<C: Cells, B: Backend> Store<C, B> {
    /// Opens an [`UnloggedScope`]. Fails with [`Error::UnloggedScopeOpen`]
    /// when one is already open.
    pub fn unlogged_scope(&self) -> Result<UnloggedScope<'_, C, B>> {
        self.open_scope()?;
        Ok(UnloggedScope {
            store: StoreRef::Borrowed(self),
            touched: HashMap::new(),
            closed: false,
        })
    }

    /// [`Self::unlogged_scope`] for a store held in an `Arc`: the scope
    /// keeps the store alive, so it can be stored beside other owners of it.
    pub fn unlogged_scope_shared(self: &Arc<Self>) -> Result<UnloggedScope<'static, C, B>> {
        self.open_scope()?;
        Ok(UnloggedScope {
            store: StoreRef::Shared(Arc::clone(self)),
            touched: HashMap::new(),
            closed: false,
        })
    }

    fn open_scope(&self) -> Result<()> {
        let mut state = self.state()?;
        state.refuse_in_scope()?;
        state.unlogged = true;
        if let Some(deps) = &state.deps {
            deps.set_unlogged(true);
        }
        Ok(())
    }
}

impl<C: Cells, B: Backend> UnloggedScope<'_, C, B> {
    /// The runtime collection behind `collection`, mutable: changes made
    /// through it never reach the log or the store's fingerprints, so the
    /// caller must undo them before closing.
    pub fn collection<T: PersistValue + Hash + Eq>(
        &self,
        collection: &PersistedCollection<T, C>,
    ) -> Result<IncrCollection<T, C>> {
        let state = self.store.state()?;
        collection.unlogged_in(&state)
    }

    /// The keyed collection behind `keyed`, writable: changes made through
    /// it never reach the log or the store's fingerprints, so the caller
    /// must undo them before closing, for instance by applying the inverses
    /// of a journal newest first.
    pub fn keyed<K, V, P>(
        &self,
        keyed: &PersistedKeyed<K, V, C, P>,
    ) -> Result<KeyedCollection<K, V, C, P, u128>>
    where
        K: Value + Hash + Eq,
        V: Value + Hash + Eq,
        P: Pair<K, V> + PersistValue,
    {
        let state = self.store.state()?;
        keyed.unlogged_in(&state)
    }

    /// The runtime node behind `input`, to set around the log; see
    /// [`touched`](Self::touched).
    pub fn input<T: PersistValue>(&self, input: Input<T>) -> Result<Incr<T>> {
        let state = self.store.state()?;
        state.binding_id(&KeyRef::from(input))?;
        Ok(input.handle)
    }

    /// Declares that the caller is about to change the input `key` refers
    /// to, which [`close`](Self::close) will then check and vouch for again
    /// when it holds its logged value. Call it before the first change; a
    /// call after one leaves the input unvouched. A collection needs no
    /// call, and one is ignored.
    pub fn touched<'k>(&mut self, key: impl Into<KeyRef<'k>>) -> Result<()> {
        let state = self.store.state()?;
        let id = state.binding_id(&key.into())?.ok_or(Error::UnknownHandle)?;
        let binding = &state.bindings[&id];
        if binding.kind() != Kind::Input {
            return Ok(());
        }
        let stamp = binding.mapped_stamp();
        let held = changed_at_erased(self.store.runtime(), binding.node());
        self.touched
            .entry(id)
            .or_insert_with(|| stamp.is_some() && stamp == held);
        Ok(())
    }

    /// Ends the scope, which requires every change to be undone. Each
    /// touched input that was unchanged when first touched and holds its
    /// logged value again is vouched for, so memo entries over it stay
    /// checkpointable. Costs one decode per touched input.
    ///
    /// A failure to compare one input leaves it unvouched and is returned
    /// after the others are handled; the scope is closed either way. A
    /// close that cannot reach the store's state, by an error or a panic,
    /// ends in the scope's drop, which poisons the store.
    pub fn close(mut self) -> Result<()> {
        let mut guard = self.store.state()?;
        let state = &mut *guard;
        let rt = self.store.runtime();
        let mut restamped = Vec::new();
        let mut failure = None;
        for (&id, &clean) in &self.touched {
            if !clean {
                continue;
            }
            let Some(binding) = state.bindings.get_mut(&id) else {
                continue;
            };
            let key = binding.key().to_owned();
            match binding.restamp(rt, state.logged.get(&key)) {
                Ok(Some((fp, stamp))) => restamped.push((key, fp, stamp)),
                Ok(None) => {}
                Err(e) => {
                    failure.get_or_insert(e);
                }
            }
        }
        state.unlogged = false;
        if let Some(deps) = state.deps.as_deref() {
            let mut map = deps.write();
            for (key, fp, stamp) in restamped {
                map.set_written(&key, fp, stamp);
            }
            drop(map);
            deps.set_unlogged(false);
        }
        self.closed = true;
        failure.map_or(Ok(()), Err)
    }
}

impl<C: Cells, B: Backend> Drop for UnloggedScope<'_, C, B> {
    fn drop(&mut self) {
        if !self.closed {
            self.store
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .poisoned = true;
        }
    }
}
