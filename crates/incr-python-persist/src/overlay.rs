//! The journal behind `Runtime.overlay()`: every mutation the binding
//! exposes records its inverse while an overlay is active, and exit applies
//! the inverses newest first and deletes the nodes the overlay created.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use incr_core::{Change, ErasedIncr, Incr, IncrCollection, KeyedCollection, PairOp, Runtime};
use incr_persist::Input;
use incr_python_protocol::{PyKey, PyValue};
use pyo3::prelude::*;

use crate::binding::PyBinding;
use crate::scope::{OverlayScope, Scope, UnloggedKeyed};
use crate::store::{Collection, Keyed};

/// A plain keyed collection of a `C` runtime.
pub type PlainKeyed<C> = KeyedCollection<PyKey, PyValue, C, PyValue, ()>;

/// Set when the overlay that created an object ends, after which the
/// object's nodes are gone and every use of it raises.
#[derive(Clone, Default)]
pub struct Born(Option<Arc<AtomicBool>>);

impl Born {
    pub fn check(&self) -> PyResult<()> {
        match &self.0 {
            Some(dead) if dead.load(Ordering::Relaxed) => {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "collection was created inside an overlay and deleted at its exit",
                ))
            }
            _ => Ok(()),
        }
    }
}

pub enum Undo<B: PyBinding> {
    Set(Incr<PyValue>, PyValue),
    Collection(IncrCollection<PyValue, B::C>, Vec<Change<PyValue>>),
    /// The pair ops a plain keyed collection's write journalled, undone
    /// by applying their inverses newest first.
    Keyed(PlainKeyed<B::C>, Vec<PairOp<PyValue, ()>>),
    /// The same for a persisted keyed collection's unlogged writes.
    PersistedKeyed(UnloggedKeyed<B>, Vec<PairOp<PyValue, u128>>),
    Label(u32, Option<String>),
}

/// An active overlay's journal: the inverses to apply, the nodes it
/// created, and the markers of the objects it made.
pub struct Overlay<B: PyBinding> {
    undo: Vec<Undo<B>>,
    set_seen: HashSet<Incr<PyValue>>,
    created: Vec<ErasedIncr>,
    dead: Vec<Arc<AtomicBool>>,
    pub reported_at_entry: HashSet<ErasedIncr>,
    persist: Option<OverlayScope<B>>,
}

impl<B: PyBinding> Default for Overlay<B> {
    fn default() -> Self {
        Overlay {
            undo: Vec::new(),
            set_seen: HashSet::new(),
            created: Vec::new(),
            dead: Vec::new(),
            reported_at_entry: HashSet::new(),
            persist: None,
        }
    }
}

fn no_scope() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("the overlay holds no persisted scope")
}

impl<B: PyBinding> Overlay<B> {
    pub fn push(&mut self, undo: Undo<B>) {
        self.undo.push(undo);
    }

    /// True the first time `node` is set in this overlay, which is the
    /// only set whose old value the exit restores.
    pub fn first_set(&mut self, node: Incr<PyValue>) -> bool {
        self.set_seen.insert(node)
    }

    pub fn forget_set(&mut self, node: Incr<PyValue>) {
        self.set_seen.remove(&node);
    }

    pub fn with_scope(&mut self, scope: Option<OverlayScope<B>>) {
        self.persist = scope;
    }

    pub fn take_scope(&mut self) -> Option<OverlayScope<B>> {
        self.persist.take()
    }

    /// Announces the first change of the persisted input `node`.
    pub fn touch(&mut self, py: Python<'_>, node: Incr<PyValue>) -> PyResult<()>
    where
        for<'a> B::Task<(&'a mut Scope<B>, Input<PyValue>)>: Send,
    {
        self.persist.as_mut().ok_or_else(no_scope)?.touch(py, node)
    }

    /// The writable runtime collection behind the persisted collection
    /// whose version node has slot `version_slot`.
    pub fn writable_collection(
        &mut self,
        py: Python<'_>,
        version_slot: u32,
    ) -> PyResult<IncrCollection<PyValue, B::C>>
    where
        for<'a> B::Task<(&'a Scope<B>, Collection<B>)>: Send,
        B::Task<IncrCollection<PyValue, B::C>>: Send,
    {
        self.persist
            .as_mut()
            .ok_or_else(no_scope)?
            .collection(py, version_slot)
    }

    pub fn writable_keyed(
        &mut self,
        py: Python<'_>,
        persisted: &Keyed<B>,
    ) -> PyResult<UnloggedKeyed<B>>
    where
        for<'a> B::Task<(&'a Scope<B>, &'a Keyed<B>)>: Send,
        B::Task<UnloggedKeyed<B>>: Send,
    {
        self.persist
            .as_ref()
            .ok_or_else(no_scope)?
            .keyed(py, persisted)
    }

    pub fn created(&mut self, node: ErasedIncr) {
        self.created.push(node);
    }

    pub fn born(&mut self) -> Born {
        let dead = Arc::new(AtomicBool::new(false));
        self.dead.push(Arc::clone(&dead));
        Born(Some(dead))
    }

    /// Applies the inverses newest first, deletes the created nodes newest
    /// first and marks their objects dead. The first failure stops the
    /// restore and is returned with the base half restored.
    pub fn restore(&mut self, rt: &Runtime<B::C>) -> PyResult<()> {
        let result = self.apply_inverses(rt);
        for node in self.created.drain(..).rev() {
            rt.delete_with_dependents(node);
        }
        result
    }

    fn apply_inverses(&mut self, rt: &Runtime<B::C>) -> PyResult<()> {
        while let Some(undo) = self.undo.pop() {
            match undo {
                Undo::Set(node, old) => rt.set(node, old),
                Undo::Collection(collection, applied) => {
                    collection.apply(rt, inverse(applied));
                }
                Undo::Keyed(keyed, ops) => {
                    keyed.apply_pairs_with(
                        rt,
                        ops.into_iter().rev().map(PairOp::inverse),
                        &mut (),
                        None,
                    );
                }
                Undo::PersistedKeyed(keyed, ops) => {
                    keyed.apply_pairs_with(
                        rt,
                        ops.into_iter().rev().map(PairOp::inverse),
                        &mut (),
                        None,
                    );
                }
                Undo::Label(slot, Some(old)) => rt.set_label(slot, old),
                Undo::Label(slot, None) => rt.clear_label(slot),
            }
        }
        Ok(())
    }
}

impl<B: PyBinding> Drop for Overlay<B> {
    fn drop(&mut self) {
        for dead in &self.dead {
            dead.store(true, Ordering::Relaxed);
        }
    }
}

/// The batch that undoes `applied`: each change flipped, last first.
pub fn inverse(applied: Vec<Change<PyValue>>) -> Vec<Change<PyValue>> {
    applied
        .into_iter()
        .rev()
        .map(|change| match change {
            Change::Insert(v) => Change::Delete(v),
            Change::Delete(v) => Change::Insert(v),
        })
        .collect()
}
