//! Keyed collections over a persisted pair collection, and the writes that
//! reach them.
//!
//! The store applies a keyed write's pairs to the collection, then the
//! index takes the keys the write staged. A record another writer
//! appended, or a compaction's change of state, reaches the collection
//! without passing `KeyedCollection`, so while a keyed collection exists
//! the store keeps one subscription that queues those records, and every
//! store call drains the queue into the indexes once it returns.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Arc, Mutex, PoisonError};

use incr_core::Local;
use incr_persist::{
    enter_value_context, Input, Origin, PersistedCollection, Record, SubscriptionId, ValueContext,
};
use incr_python_protocol::PyValue;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::store::StoreCore;
use crate::keyed::{KeyCounts, KeyIndex, SharedIndex, StagedKeys};
use crate::{index_in_use, PyCollection, PyKeyedCollection, PyNodeId};
use incr_compute::Change;

pub(super) type PyPersistedCollection = PersistedCollection<PyValue, Local>;

type Queue = Arc<Mutex<Vec<(Record, Origin)>>>;

/// A persisted pair collection and the index over its pairs.
pub(super) struct KeyedBinding {
    key: String,
    pub(super) pairs: PyPersistedCollection,
    pub(super) index: Arc<SharedIndex>,
    /// Counts the changes to the index that did not come from this
    /// binding's own writes, so a write can tell that what it staged came
    /// from an index that has since moved.
    epoch: Cell<u64>,
}

impl KeyedBinding {
    fn moved(&self) {
        self.epoch.set(self.epoch.get() + 1);
    }
}

/// How often a write recomputes its staged keys against the collection
/// before giving up, when writers keep changing the same keys.
const REPAIR_ROUNDS: usize = 3;

/// The store subscription that queues records the store did not write.
pub(super) struct Following {
    id: SubscriptionId,
    queue: Queue,
}

enum Op {
    Set(Input<PyValue>, PyValue),
    Insert(PyPersistedCollection, PyValue),
    Delete(PyPersistedCollection, PyValue),
    Replace(PyPersistedCollection, Option<PyValue>, PyValue),
}

struct Staging {
    binding: Rc<KeyedBinding>,
    staged: StagedKeys,
    epoch: u64,
}

/// The writes of one store call or batch, collected before the store
/// appends them as one run. Keyed writes are staged per collection, in a
/// map the committed index is consulted behind, so several writes to one
/// key compose.
#[derive(Default)]
pub(super) struct Pending {
    ops: Vec<Op>,
    keyed: Vec<Staging>,
}

impl Pending {
    pub(super) fn set(
        &mut self,
        core: &StoreCore,
        node: &PyNodeId,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        self.ops.push(Op::Set(core.input_of(node)?, PyValue(value)));
        Ok(())
    }

    pub(super) fn insert(
        &mut self,
        core: &StoreCore,
        collection: &PyCollection,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.ops
            .push(Op::Insert(core.collection_of(collection)?, PyValue(row)));
        Ok(())
    }

    pub(super) fn delete(
        &mut self,
        core: &StoreCore,
        collection: &PyCollection,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.ops
            .push(Op::Delete(core.collection_of(collection)?, PyValue(row)));
        Ok(())
    }

    pub(super) fn replace(
        &mut self,
        core: &StoreCore,
        collection: &PyCollection,
        old: Py<PyAny>,
        new: Py<PyAny>,
    ) -> PyResult<()> {
        self.ops.push(Op::Replace(
            core.collection_of(collection)?,
            Some(PyValue(old)),
            PyValue(new),
        ));
        Ok(())
    }

    fn staging(&mut self, binding: Rc<KeyedBinding>) -> &mut StagedKeys {
        let at = match self
            .keyed
            .iter()
            .position(|s| Rc::ptr_eq(&s.binding, &binding))
        {
            Some(at) => at,
            None => {
                self.keyed.push(Staging {
                    epoch: binding.epoch.get(),
                    binding,
                    staged: StagedKeys::default(),
                });
                self.keyed.len() - 1
            }
        };
        &mut self.keyed[at].staged
    }

    /// Stages `f` over the index of `kc`'s binding, with the index held as
    /// a keyed write holds it while user key and row code runs.
    fn stage(
        &mut self,
        core: &StoreCore,
        kc: &PyKeyedCollection,
        f: impl FnOnce(&KeyIndex, &mut StagedKeys) -> PyResult<()>,
    ) -> PyResult<()> {
        let binding = core.keyed_of(kc)?;
        let index = Arc::clone(&binding.index);
        let guard = index
            .try_borrow_for_write(core.runtime().hold_routing_suppressed())
            .ok_or_else(index_in_use)?;
        f(&guard, self.staging(binding))
    }

    pub(super) fn upsert(
        &mut self,
        core: &StoreCore,
        kc: &PyKeyedCollection,
        key: &Bound<'_, PyAny>,
        row: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.stage(core, kc, |index, staged| {
            staged.stage(index, key, Some(row)).map(drop)
        })
    }

    pub(super) fn remove(
        &mut self,
        core: &StoreCore,
        kc: &PyKeyedCollection,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.stage(core, kc, |index, staged| {
            if staged.stage(index, key, None)? {
                Ok(())
            } else {
                Err(PyKeyError::new_err(key.clone().unbind()))
            }
        })
    }

    pub(super) fn reconcile(
        &mut self,
        core: &StoreCore,
        kc: &PyKeyedCollection,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.stage(core, kc, |index, staged| {
            staged.stage_reconcile(index, rows)
        })
    }

    /// What the first keyed collection's staged keys insert, replace and
    /// remove.
    pub(super) fn counts(&self) -> KeyCounts {
        self.keyed
            .first()
            .map_or((0, 0, 0), |staging| staging.staged.counts())
    }
}

fn pair_of(value: PyValue, py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
    Ok(value.0.into_bound(py).downcast_into::<PyTuple>()?)
}

impl StoreCore {
    pub(super) fn keyed_of(&self, kc: &PyKeyedCollection) -> PyResult<Rc<KeyedBinding>> {
        self.keyed
            .borrow()
            .iter()
            .find(|binding| binding.pairs.handle().version_node() == kc.pairs.version_node())
            .cloned()
            .ok_or_else(|| {
                PyValueError::new_err(
                    "collection is not a persisted keyed collection of this store",
                )
            })
    }

    /// Appends `pending` as one run, applies it, then lets each staged
    /// index take its keys. An index that moved since its keys were staged,
    /// because the call also took records another writer appended, is
    /// rebuilt and the keys are repaired instead.
    pub(super) fn write(&self, py: Python<'_>, pending: Pending) -> PyResult<()> {
        let Pending { ops, keyed } = pending;
        let changes: Vec<Vec<Change<PyValue>>> =
            keyed.iter().map(|s| s.staged.changes(py)).collect();
        self.run(py, |store| {
            store
                .batch(|batch| {
                    for op in ops {
                        match op {
                            Op::Set(input, value) => batch.set(input, value),
                            Op::Insert(collection, row) => batch.insert(&collection, row),
                            Op::Delete(collection, row) => batch.delete(&collection, &row),
                            Op::Replace(collection, old, new) => {
                                batch.replace(&collection, old.as_ref(), new)
                            }
                        };
                    }
                    for (staging, changes) in keyed.iter().zip(changes) {
                        for change in changes {
                            match change {
                                Change::Insert(pair) => {
                                    batch.insert(&staging.binding.pairs, pair);
                                }
                                Change::Delete(pair) => {
                                    batch.delete(&staging.binding.pairs, &pair);
                                }
                            }
                        }
                    }
                })
                .map(drop)
        })?;
        for Staging {
            binding,
            staged,
            epoch,
        } in &keyed
        {
            if binding.epoch.get() != *epoch {
                self.repair(py, binding, staged)?;
            } else {
                let mut index = binding.index.try_borrow_mut().ok_or_else(index_in_use)?;
                index.commit_staged(py, staged);
            }
        }
        Ok(())
    }

    /// Leaves each key `staged` altered holding the pair it was staged to
    /// hold: the write was computed from an index that lacked records the
    /// collection took in the same call, so the collection may hold other
    /// pairs for those keys, which a further write deletes. The staged
    /// write is the later one, so it wins, and both stores end with one
    /// pair per key once the other has refreshed.
    fn repair(&self, py: Python<'_>, binding: &KeyedBinding, staged: &StagedKeys) -> PyResult<()> {
        for _ in 0..REPAIR_ROUNDS {
            self.rebuild(py, binding)?;
            let fixes = binding
                .index
                .try_borrow_mut()
                .ok_or_else(index_in_use)?
                .repairs(py, staged)?;
            if fixes.is_empty() {
                return Ok(());
            }
            self.run(py, |store| {
                store
                    .batch(|batch| {
                        for fix in fixes {
                            match fix {
                                Change::Insert(pair) => batch.insert(&binding.pairs, pair),
                                Change::Delete(pair) => batch.delete(&binding.pairs, &pair),
                            };
                        }
                    })
                    .map(drop)
            })?;
        }
        Err(PyRuntimeError::new_err(format!(
            "keyed collection {:?} kept changing under other writers: refresh and retry",
            binding.key
        )))
    }

    /// Declares the persisted collection `key` and the index over its
    /// restored pairs, subscribing to the store first if no keyed
    /// collection has yet.
    pub(super) fn declare_keyed(&self, py: Python<'_>, key: &str) -> PyResult<Rc<KeyedBinding>> {
        let subscribed = self.following.borrow().is_none();
        if subscribed {
            self.follow(py)?;
        }
        let declared = self.index_restored_pairs(py, key);
        match &declared {
            Ok(binding) => self.keyed.borrow_mut().push(Rc::clone(binding)),
            Err(_) if subscribed => self.unfollow(py)?,
            Err(_) => {}
        }
        declared
    }

    fn index_restored_pairs(&self, py: Python<'_>, key: &str) -> PyResult<Rc<KeyedBinding>> {
        let pairs = self.run(py, |store| store.collection::<PyValue>(key))?;
        let index = Arc::new(SharedIndex::holding(self.index_of(py, &pairs)?));
        Ok(Rc::new(KeyedBinding {
            key: key.to_owned(),
            pairs,
            index,
            epoch: Cell::new(0),
        }))
    }

    fn index_of(&self, py: Python<'_>, pairs: &PyPersistedCollection) -> PyResult<KeyIndex> {
        let held = pairs.handle().values(self.runtime());
        let tuples = held
            .into_iter()
            .map(|pair| pair_of(pair, py))
            .collect::<PyResult<Vec<_>>>()?;
        KeyIndex::of_pairs(tuples)
    }

    fn rebuild(&self, py: Python<'_>, binding: &KeyedBinding) -> PyResult<()> {
        let fresh = self.index_of(py, &binding.pairs)?;
        let mut index = binding.index.try_borrow_mut().ok_or_else(index_in_use)?;
        *index = fresh;
        binding.moved();
        Ok(())
    }

    fn follow(&self, py: Python<'_>) -> PyResult<()> {
        let queue = Queue::default();
        let pushed = Arc::clone(&queue);
        let id = self.run(py, |store| {
            store.subscribe(move |_, record, origin| {
                if origin != Origin::Local {
                    pushed
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push((record.clone(), origin));
                }
            })
        })?;
        *self.following.borrow_mut() = Some(Following { id, queue });
        Ok(())
    }

    fn unfollow(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(following) = self.following.borrow_mut().take() {
            self.run(py, |store| store.unsubscribe(following.id))?;
        }
        Ok(())
    }

    /// Applies the records queued since the last drain to the keyed
    /// collections they name. A key that saw a `Delete` or `Forget`, or
    /// whose records fail to apply, has its index rebuilt from the
    /// collection, which the store already brought up to date.
    pub(super) fn drain(&self, py: Python<'_>) -> PyResult<()> {
        let records = match &*self.following.borrow() {
            Some(following) => std::mem::take(
                &mut *following
                    .queue
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner),
            ),
            None => return Ok(()),
        };
        if records.is_empty() {
            return Ok(());
        }
        let bindings = self.keyed.borrow().clone();
        for binding in &bindings {
            let mine: Vec<&Record> = records
                .iter()
                .map(|(record, _)| record)
                .filter(|record| record.key() == binding.key)
                .collect();
            if mine.is_empty() {
                continue;
            }
            binding.moved();
            let resets = mine
                .iter()
                .any(|record| matches!(record, Record::Delete { .. } | Record::Forget { .. }));
            if resets || self.apply_foreign(py, binding, &mine).is_err() {
                self.rebuild(py, binding)?;
            }
        }
        Ok(())
    }

    fn apply_foreign(
        &self,
        py: Python<'_>,
        binding: &KeyedBinding,
        records: &[&Record],
    ) -> PyResult<()> {
        let mut index = binding.index.try_borrow_mut().ok_or_else(index_in_use)?;
        for record in records {
            match record {
                Record::Insert { element, .. } => {
                    index.insert_pair(self.decode_pair(py, &binding.key, element)?)?;
                }
                Record::Remove { element, .. } => {
                    index.remove_pair(&self.decode_pair(py, &binding.key, element)?)?;
                }
                Record::Replace { old, new, .. } => {
                    if let Some(old) = old {
                        index.remove_pair(&self.decode_pair(py, &binding.key, old)?)?;
                    }
                    index.insert_pair(self.decode_pair(py, &binding.key, new)?)?;
                }
                Record::Declare { .. }
                | Record::Set { .. }
                | Record::Delete { .. }
                | Record::Forget { .. } => {}
            }
        }
        Ok(())
    }

    fn decode_pair<'py>(
        &self,
        py: Python<'py>,
        key: &str,
        element: &[u8],
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _context = enter_value_context(ValueContext::Key(key.to_owned()));
        let decoded = postcard::take_from_bytes::<PyValue>(element);
        match decoded {
            Ok((pair, [])) => pair_of(pair, py),
            Ok(_) => Err(PyValueError::new_err("a logged pair has trailing bytes")),
            Err(e) => Err(self
                .registry
                .take_failure()
                .unwrap_or_else(|| PyValueError::new_err(e.to_string()))),
        }
    }
}
