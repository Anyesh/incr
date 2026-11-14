//! The unlogged scope an overlay holds on a persisted graph: its changes to
//! persisted inputs and collections reach the runtime only, so exit
//! restores them without the log or the memo ever seeing a candidate.

use std::collections::HashMap;
use std::rc::Rc;

use incr_compute::IncrCollection;
use incr_core::Local;
use incr_persist::UnloggedScope;
use incr_python_protocol::PyValue;
use pyo3::prelude::*;

use super::keyed::PyPersistedKeyed;
use super::persist_error;
use super::store::{PyPersistedCollection, StoreCore};
use super::PyBackend;
use crate::{PersistedCore, PyCollection, PyNodeId};

/// An open [`UnloggedScope`] with the writable handles it gave out.
/// Dropping it without [`Self::close`] poisons the store.
pub(crate) struct OverlayScope {
    scope: UnloggedScope<'static, Local, PyBackend>,
    core: Rc<StoreCore>,
    handles: HashMap<u32, IncrCollection<PyValue>>,
}

impl OverlayScope {
    pub(crate) fn open(py: Python<'_>, core: &Rc<StoreCore>) -> PyResult<Self> {
        let scope = core
            .store
            .unlogged_scope_shared()
            .map_err(|e| persist_error(py, e))?;
        Ok(Self {
            scope,
            core: Rc::clone(core),
            handles: HashMap::new(),
        })
    }

    /// Announces that `node` is about to change, before its first change.
    pub(crate) fn touch(&mut self, node: &PyNodeId) -> PyResult<()> {
        let input = self.core.input_of(node)?;
        self.scope
            .touched(input)
            .map_err(|e| Python::attach(|py| persist_error(py, e)))
    }

    pub(crate) fn collection(
        &mut self,
        collection: &PyCollection,
    ) -> PyResult<IncrCollection<PyValue>> {
        let persisted = self.core.collection_of(collection)?;
        self.writable(&persisted)
    }

    pub(crate) fn keyed(&self, persisted: &PyPersistedKeyed) -> PyResult<PersistedCore> {
        self.scope
            .keyed(persisted)
            .map_err(|e| Python::attach(|py| persist_error(py, e)))
    }

    fn writable(&mut self, persisted: &PyPersistedCollection) -> PyResult<IncrCollection<PyValue>> {
        let slot = persisted.handle().version_node().slot();
        if let Some(handle) = self.handles.get(&slot) {
            return Ok(handle.clone());
        }
        let handle = self
            .scope
            .collection(persisted)
            .map_err(|e| Python::attach(|py| persist_error(py, e)))?;
        self.handles.insert(slot, handle.clone());
        Ok(handle)
    }

    /// Ends the scope after the overlay restored every change, vouching
    /// for the memo entries over the inputs it touched.
    pub(crate) fn close(self, py: Python<'_>) -> PyResult<()> {
        let Self { scope, core, .. } = self;
        core.registry.run(py, || scope.close())
    }
}
