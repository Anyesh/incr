//! The unlogged scope an overlay holds on a persisted graph: its changes to
//! persisted inputs and collections reach the runtime only, so exit
//! restores them without the log or the memo ever seeing a candidate.

use std::collections::HashMap;
use std::sync::Arc;

use incr_core::{Incr, IncrCollection, KeyedCollection};
use incr_persist::UnloggedScope;
use incr_python_protocol::{PyKey, PyValue};
use pyo3::prelude::*;

use crate::backend::PyBackend;
use crate::binding::PyBinding;
use crate::persist_error;
use crate::store::{Keyed, StoreCore};

/// The writable runtime collection behind a persisted keyed collection.
pub type UnloggedKeyed<B> = KeyedCollection<PyKey, PyValue, <B as PyBinding>::C, PyValue, u128>;

/// An open [`UnloggedScope`] with the writable handles it gave out.
/// Dropping it without [`Self::close`] poisons the store.
pub struct OverlayScope<B: PyBinding> {
    scope: UnloggedScope<'static, B::C, PyBackend<B::Exceptions>>,
    core: Arc<StoreCore<B>>,
    handles: HashMap<u32, IncrCollection<PyValue, B::C>>,
}

impl<B: PyBinding> OverlayScope<B> {
    pub fn open(py: Python<'_>, core: &Arc<StoreCore<B>>) -> PyResult<Self> {
        let scope = core
            .store()
            .unlogged_scope_shared()
            .map_err(|e| persist_error::<B::Exceptions>(py, e))?;
        Ok(Self {
            scope,
            core: Arc::clone(core),
            handles: HashMap::new(),
        })
    }

    /// Announces that the persisted input `node` is about to change, before
    /// its first change.
    pub fn touch(&mut self, node: Incr<PyValue>) -> PyResult<()> {
        let input = self.core.input_of(node)?;
        self.scope
            .touched(input)
            .map_err(|e| Python::attach(|py| persist_error::<B::Exceptions>(py, e)))
    }

    /// The writable runtime collection behind the persisted collection
    /// whose version node has slot `version_slot`.
    pub fn collection(&mut self, version_slot: u32) -> PyResult<IncrCollection<PyValue, B::C>> {
        if let Some(handle) = self.handles.get(&version_slot) {
            return Ok(handle.clone());
        }
        let persisted = self.core.collection_of(version_slot)?;
        let handle = self
            .scope
            .collection(&persisted)
            .map_err(|e| Python::attach(|py| persist_error::<B::Exceptions>(py, e)))?;
        self.handles.insert(version_slot, handle.clone());
        Ok(handle)
    }

    pub fn keyed(&self, persisted: &Keyed<B>) -> PyResult<UnloggedKeyed<B>> {
        self.scope
            .keyed(persisted)
            .map_err(|e| Python::attach(|py| persist_error::<B::Exceptions>(py, e)))
    }

    /// Ends the scope after the overlay restored every change, vouching
    /// for the memo entries over the inputs it touched.
    pub fn close(self, py: Python<'_>) -> PyResult<()> {
        let Self { scope, core, .. } = self;
        core.registry().run(py, || scope.close())
    }
}
