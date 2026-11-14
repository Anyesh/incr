//! The unlogged scope an overlay holds on a persisted graph: its changes to
//! persisted inputs and collections reach the runtime only, so exit
//! restores them without the log or the memo ever seeing a candidate.

use std::collections::HashMap;
use std::sync::Arc;

use incr_core::{Incr, IncrCollection, KeyedCollection};
use incr_persist::{Input, UnloggedScope};
use incr_python_protocol::{PyKey, PyValue};
use pyo3::prelude::*;

use crate::backend::PyBackend;
use crate::binding::{carry, Carry, PyBinding};
use crate::store::{Collection, Keyed, StoreCore};

/// The writable runtime collection behind a persisted keyed collection.
pub type UnloggedKeyed<B> = KeyedCollection<PyKey, PyValue, <B as PyBinding>::C, PyValue, u128>;

/// The unlogged scope of a package's store.
pub type Scope<B> =
    UnloggedScope<'static, <B as PyBinding>::C, PyBackend<<B as PyBinding>::Exceptions>>;

/// An open [`UnloggedScope`] with the writable handles it gave out.
/// Dropping it without [`Self::close`] poisons the store. Every call that
/// takes the store mutex runs through [`StoreCore::write`], so a Shared
/// one releases the GIL.
pub struct OverlayScope<B: PyBinding> {
    scope: Scope<B>,
    core: Arc<StoreCore<B>>,
    handles: HashMap<u32, IncrCollection<PyValue, B::C>>,
}

impl<B: PyBinding> OverlayScope<B> {
    pub fn open(py: Python<'_>, core: &Arc<StoreCore<B>>) -> PyResult<Self>
    where
        B::Task<Scope<B>>: Send,
    {
        let scope = core.write(py, |store, _| {
            store.unlogged_scope_shared().map(carry::<B, _>)
        })?;
        Ok(Self {
            scope: scope.into_inner(),
            core: Arc::clone(core),
            handles: HashMap::new(),
        })
    }

    /// Announces that the persisted input `node` is about to change, before
    /// its first change.
    pub fn touch(&mut self, py: Python<'_>, node: Incr<PyValue>) -> PyResult<()>
    where
        for<'a> B::Task<(&'a mut Scope<B>, Input<PyValue>)>: Send,
    {
        let input = self.core.input_of(node)?;
        let task = carry::<B, _>((&mut self.scope, input));
        self.core.write(py, move |_, _| {
            let (scope, input) = task.into_inner();
            scope.touched(input)
        })
    }

    /// The writable runtime collection behind the persisted collection
    /// whose version node has slot `version_slot`.
    pub fn collection(
        &mut self,
        py: Python<'_>,
        version_slot: u32,
    ) -> PyResult<IncrCollection<PyValue, B::C>>
    where
        for<'a> B::Task<(&'a Scope<B>, Collection<B>)>: Send,
        B::Task<IncrCollection<PyValue, B::C>>: Send,
    {
        if let Some(handle) = self.handles.get(&version_slot) {
            return Ok(handle.clone());
        }
        let persisted = self.core.collection_of(version_slot)?;
        let task = carry::<B, _>((&self.scope, persisted));
        let core = &self.core;
        let handle = core.write(py, move |_, _| {
            let (scope, persisted) = task.into_inner();
            scope.collection(&persisted).map(carry::<B, _>)
        })?;
        let handle = handle.into_inner();
        self.handles.insert(version_slot, handle.clone());
        Ok(handle)
    }

    pub fn keyed(&self, py: Python<'_>, persisted: &Keyed<B>) -> PyResult<UnloggedKeyed<B>>
    where
        for<'a> B::Task<(&'a Scope<B>, &'a Keyed<B>)>: Send,
        B::Task<UnloggedKeyed<B>>: Send,
    {
        let task = carry::<B, _>((&self.scope, persisted));
        let core = &self.core;
        let keyed = core.write(py, move |_, _| {
            let (scope, persisted) = task.into_inner();
            scope.keyed(persisted).map(carry::<B, _>)
        })?;
        Ok(keyed.into_inner())
    }

    /// Ends the scope after the overlay restored every change, vouching
    /// for the memo entries over the inputs it touched.
    pub fn close(self, py: Python<'_>) -> PyResult<()>
    where
        B::Task<Scope<B>>: Send,
    {
        let Self { scope, core, .. } = self;
        let scope = carry::<B, _>(scope);
        core.write(py, move |_, _| scope.into_inner().close())
    }
}
