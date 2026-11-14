//! A store batch: the writes of a `with` block, either collected and
//! appended as one run when the block ends ([`Apply::OnExit`]), or applied
//! as they are made through a staged batch that commits when the block
//! ends ([`Apply::OnWrite`]).

use incr_core::Incr;
use incr_persist::{Error, PendingWrite, StagedBatch};
use incr_python_protocol::{PyKey, PyValue};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::backend::PyBackend;
use crate::binding::{carry, Carry, PyBinding};
use crate::keyed::{Pending, Staging};
use crate::persist_error;
use crate::store::{Collection, Keyed, StoreCore};

/// The staged batch an `on_write` batch writes through.
pub type Staged<B> =
    StagedBatch<'static, <B as PyBinding>::C, PyBackend<<B as PyBinding>::Exceptions>>;

/// When a batch's writes reach the runtime.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Apply {
    /// Collected, then appended and applied as one run when the block ends
    /// without an exception.
    OnExit,
    /// Applied at once, appended as one run when the block ends.
    OnWrite,
}

impl Apply {
    /// `"on_exit"` or `"on_write"`; `ValueError` otherwise.
    pub fn parse(apply: &str) -> PyResult<Self> {
        match apply {
            "on_exit" => Ok(Apply::OnExit),
            "on_write" => Ok(Apply::OnWrite),
            other => Err(PyValueError::new_err(format!(
                "apply must be 'on_exit' or 'on_write', not {other:?}"
            ))),
        }
    }
}

#[derive(PartialEq)]
enum Phase {
    New,
    Open,
    Done,
}

enum Writes<B: PyBinding> {
    Collected(Pending<B>),
    /// The staged batch while it is open or awaits a retry; `None` before
    /// the block, once its run landed, and once the block raised.
    Staged(Option<Staged<B>>),
}

/// A batch's phase and its writes.
pub struct Batch<B: PyBinding> {
    phase: Phase,
    raised: bool,
    writes: Writes<B>,
}

/// One write of an `on_write` batch, carried to its staged batch.
pub enum Op<B: PyBinding> {
    Set(incr_persist::Input<PyValue>, PyValue),
    Insert(Collection<B>, PyValue),
    Delete(Collection<B>, PyValue),
    Replace(Collection<B>, PyValue, PyValue),
    Pending(PendingWrite<B::C>),
}

impl<B: PyBinding> Op<B> {
    fn apply(self, staged: &mut Staged<B>) -> incr_persist::Result<()> {
        match self {
            Op::Set(input, value) => staged.set(input, value),
            Op::Insert(collection, row) => staged.insert(&collection, row),
            Op::Delete(collection, row) => staged.delete(&collection, row),
            Op::Replace(collection, old, new) => staged.replace(&collection, Some(old), new),
            Op::Pending(pending) => staged.write_pending(pending),
        }
    }
}

fn not_open() -> PyErr {
    PyRuntimeError::new_err("the batch is not open: use it as `with store.batch() as b`")
}

impl<B: PyBinding> Batch<B> {
    pub fn new(apply: Apply) -> Self {
        Batch {
            phase: Phase::New,
            raised: false,
            writes: match apply {
                Apply::OnExit => Writes::Collected(Pending::default()),
                Apply::OnWrite => Writes::Staged(None),
            },
        }
    }

    fn require_open(&self) -> PyResult<()> {
        if self.phase == Phase::Open {
            Ok(())
        } else {
            Err(not_open())
        }
    }

    /// The collected writes of an open `on_exit` batch, or `None` for an
    /// open `on_write` one; `RuntimeError` outside the block.
    fn collected(&mut self) -> PyResult<Option<&mut Pending<B>>> {
        self.require_open()?;
        Ok(match &mut self.writes {
            Writes::Collected(pending) => Some(pending),
            Writes::Staged(_) => None,
        })
    }

    /// Ends the block after it raised: an `on_write` batch drops its staged
    /// batch, which poisons the store, since its writes are in the runtime
    /// and never reach the log. The drop goes through the binding's
    /// `drop_detached` and no engine call, so it never raises a `group_by`
    /// routing failure an earlier call stored in place of the block's own
    /// exception; the failure stays for the next engine call, as on a
    /// Local runtime.
    pub fn abandon(&mut self, py: Python<'_>) -> PyResult<()>
    where
        B::Task<Staged<B>>: Send,
    {
        self.require_open()?;
        self.phase = Phase::Done;
        self.raised = true;
        match &mut self.writes {
            Writes::Collected(pending) => drop(std::mem::take(pending)),
            Writes::Staged(staged) => {
                if let Some(batch) = staged.take() {
                    B::drop_detached(py, batch);
                }
            }
        }
        Ok(())
    }
}

impl<B: PyBinding> Batch<B>
where
    B::Task<Staged<B>>: Send,
    B::Task<PendingWrite<B::C>>: Send,
    for<'a> B::Task<(&'a mut Staged<B>, Op<B>)>: Send,
    for<'a> B::Task<&'a mut Staged<B>>: Send,
{
    /// Opens the batch; `RuntimeError` if it was entered before. An
    /// `on_write` batch opens its staged batch on `core`'s store, which
    /// catches up, runs a due auto-checkpoint and fails as
    /// `Store::staged_batch` does.
    pub fn enter(&mut self, py: Python<'_>, core: &StoreCore<B>) -> PyResult<()> {
        if self.phase != Phase::New {
            return Err(PyRuntimeError::new_err("a batch can be entered once"));
        }
        if let Writes::Staged(staged) = &mut self.writes {
            let opened = core.write(py, |store, _| {
                store.staged_batch_shared().map(carry::<B, _>)
            })?;
            *staged = Some(opened.into_inner());
        }
        self.phase = Phase::Open;
        Ok(())
    }

    /// Ends the block, which did not raise: an `on_exit` batch appends and
    /// applies its writes; an `on_write` batch commits, keeping its run for
    /// [`Self::retry`] when the commit fails.
    pub fn exit(&mut self, py: Python<'_>, core: &StoreCore<B>) -> PyResult<()> {
        self.require_open()?;
        self.phase = Phase::Done;
        match &mut self.writes {
            Writes::Collected(pending) => core.write_pending(py, std::mem::take(pending)),
            Writes::Staged(_) => self.land(py, core, false).map(drop),
        }
    }

    /// Attempts an `on_write` batch's failed commit again, returning the
    /// last position of its run. `RuntimeError` for an `on_exit` batch, an
    /// open one or one whose block raised; `PersistError` once its run has
    /// landed.
    pub fn retry(&mut self, py: Python<'_>, core: &StoreCore<B>) -> PyResult<u64> {
        let retriable = match &self.writes {
            Writes::Collected(_) => false,
            Writes::Staged(_) => self.phase == Phase::Done && !self.raised,
        };
        if !retriable {
            return Err(PyRuntimeError::new_err(
                "only an on_write batch whose commit failed can retry, after its block",
            ));
        }
        self.land(py, core, true)
    }

    fn land(&mut self, py: Python<'_>, core: &StoreCore<B>, retry: bool) -> PyResult<u64> {
        let Writes::Staged(staged) = &mut self.writes else {
            unreachable!("only an on_write batch lands a staged run")
        };
        let Some(batch) = staged.as_mut() else {
            return Err(persist_error::<B::Exceptions>(py, Error::StagedBatchClosed));
        };
        let batch = carry::<B, _>(batch);
        let landed = core.write(py, move |_, _| {
            let batch = batch.into_inner();
            if retry {
                batch.retry()
            } else {
                batch.commit()
            }
        })?;
        *staged = None;
        Ok(landed.0)
    }

    fn write_staged(&mut self, py: Python<'_>, core: &StoreCore<B>, op: Op<B>) -> PyResult<()> {
        self.require_open()?;
        let Writes::Staged(Some(staged)) = &mut self.writes else {
            unreachable!("an open on_write batch holds its staged batch")
        };
        let task = carry::<B, _>((staged, op));
        core.write(py, move |_, _| {
            let (staged, op) = task.into_inner();
            op.apply(staged)
        })
    }

    /// Writes the persisted input `node` of `core`.
    pub fn set(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        node: Incr<PyValue>,
        value: Py<PyAny>,
    ) -> PyResult<()> {
        match self.collected()? {
            Some(pending) => pending.set(core, node, value),
            None => {
                let input = core.input_of(node)?;
                self.write_staged(py, core, Op::Set(input, PyValue(value)))
            }
        }
    }

    /// Appends `row` to the collection whose version node has slot
    /// `version_slot`.
    pub fn insert(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        version_slot: u32,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        match self.collected()? {
            Some(pending) => pending.insert(core, version_slot, row),
            None => {
                let collection = core.collection_of(version_slot)?;
                self.write_staged(py, core, Op::Insert(collection, PyValue(row)))
            }
        }
    }

    /// Removes one occurrence of `row`.
    pub fn delete(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        version_slot: u32,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        match self.collected()? {
            Some(pending) => pending.delete(core, version_slot, row),
            None => {
                let collection = core.collection_of(version_slot)?;
                self.write_staged(py, core, Op::Delete(collection, PyValue(row)))
            }
        }
    }

    /// Replaces one occurrence of `old`, if present, with `new`.
    pub fn replace(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        version_slot: u32,
        old: Py<PyAny>,
        new: Py<PyAny>,
    ) -> PyResult<()> {
        match self.collected()? {
            Some(pending) => pending.replace(core, version_slot, old, new),
            None => {
                let collection = core.collection_of(version_slot)?;
                let op = Op::Replace(collection, PyValue(old), PyValue(new));
                self.write_staged(py, core, op)
            }
        }
    }
}

impl<B: PyBinding> Batch<B>
where
    B::Task<Staged<B>>: Send,
    B::Task<PendingWrite<B::C>>: Send,
    for<'a> B::Task<(&'a mut Staged<B>, Op<B>)>: Send,
    for<'a> B::Task<&'a mut Staged<B>>: Send,
    for<'a> B::Task<Staging<'a, B>>: Send,
{
    /// Runs `stage` on the open `on_exit` batch's writes, or on fresh ones
    /// an `on_write` batch then applies at once; a key's staged rows are
    /// read from the index, which holds every earlier `on_write` write.
    fn keyed(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        stage: impl FnOnce(&mut Pending<B>) -> PyResult<()>,
    ) -> PyResult<()> {
        match self.collected()? {
            Some(pending) => stage(pending),
            None => {
                let mut pending = Pending::default();
                stage(&mut pending)?;
                self.write_staged(py, core, Op::Pending(pending.into_write()))
            }
        }
    }

    /// Holds `row` for `key` in `keyed`.
    pub fn upsert(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        keyed: &Keyed<B>,
        key: Py<PyAny>,
        row: Py<PyAny>,
    ) -> PyResult<()> {
        self.keyed(py, core, |pending| {
            pending.upsert(py, core, keyed, key, row)
        })
    }

    /// Removes `key` from `keyed`, raising `KeyError` when it holds no row
    /// as written so far.
    pub fn remove(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        keyed: &Keyed<B>,
        key: Py<PyAny>,
    ) -> PyResult<()> {
        self.keyed(py, core, |pending| pending.remove(py, core, keyed, key))
    }

    /// Makes `keyed` hold exactly `rows`.
    pub fn reconcile(
        &mut self,
        py: Python<'_>,
        core: &StoreCore<B>,
        keyed: &Keyed<B>,
        rows: Vec<(PyKey, PyValue)>,
    ) -> PyResult<()> {
        self.keyed(py, core, |pending| pending.reconcile(py, core, keyed, rows))
    }
}
