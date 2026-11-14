use std::hash::Hash;
use std::sync::{Arc, PoisonError};

use incr_core::{Cells, Runtime};

use super::scope::StoreRef;
use super::{
    set_record, Durability, Input, Origin, PendingWrite, PersistValue, PersistedCollection, State,
    Store,
};
use crate::{Backend, Error, Position, ReadOutcome, Record, Result};

/// Where a staged batch calls [`STAGED_PARK`] on its thread, store mutex
/// held.
#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StagedPark {
    /// One staged record is applied to the runtime and kept for the run.
    Applied,
    /// The commit's run is in the log and the store's position has not
    /// moved yet.
    Landed,
}

#[cfg(test)]
type StagedParkHook = Box<dyn Fn(StagedPark)>;

#[cfg(test)]
thread_local! {
    pub(crate) static STAGED_PARK: std::cell::RefCell<Option<StagedParkHook>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn park(point: StagedPark) {
    STAGED_PARK.with(|hook| {
        if let Some(hook) = &*hook.borrow() {
            hook(point);
        }
    });
}

/// Writes applied to the runtime as they are made and appended to the log
/// as one run at [`commit`](Self::commit), opened by
/// [`Store::staged_batch`].
///
/// Each write builds its record, applies it through the store's normal
/// apply path at once, so queries read it before the commit, and keeps the
/// record; the store's position does not move until the commit. While the
/// batch is open the store refuses every other call that writes, declares
/// or binds a key, reads the log, checkpoints (the store's or a memo's) or
/// opens an unlogged scope, with [`Error::StagedBatchOpen`]. On a `Shared`
/// runtime other threads see the staged writes before they are logged.
///
/// The commit lands the run with
/// [`ConditionalAppend::append_at`](crate::ConditionalAppend::append_at)
/// right after the position the store had applied, so a run from an
/// attempt that failed ambiguously and its retry can never both land.
/// Subscribers hear each record at the commit, as [`Origin::Local`].
///
/// The store is poisoned, since its runtime may then hold values the log
/// never receives, when a staged record fails to apply, when the run is
/// found behind another writer's records, when a failure follows the
/// run's landing, and when the batch is dropped without a landed commit,
/// by a panic inside the block included.
#[must_use = "dropping a staged batch without committing it poisons the store"]
pub struct StagedBatch<'s, C: Cells, B: Backend> {
    store: StoreRef<'s, C, B>,
    run: Vec<Record>,
    phase: Phase,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Open,
    AwaitingRetry,
    Landed,
}

/// How a commit attempt failed.
enum Failure {
    /// The run may or may not have landed; a retry finds out.
    Kept(Error),
    /// The store can no longer match its runtime to the log.
    Poison(Error),
}

impl<C: Cells, B: Backend> Store<C, B> {
    /// Opens a [`StagedBatch`]: catches up with the log, runs the
    /// checkpoint [`Options::auto_checkpoint_records`](super::Options)
    /// asks for if it is due, so that none falls between the batch's
    /// records, and marks the store staged.
    ///
    /// Fails with [`Error::ConditionalAppendUnsupported`] when the backend
    /// offers no conditional append, [`Error::ReadOnly`] on a read-only
    /// backend, [`Error::StagedBatchOpen`] or [`Error::UnloggedScopeOpen`]
    /// while a batch or a scope is open, and as a catch-up or checkpoint
    /// does.
    pub fn staged_batch(&self) -> Result<StagedBatch<'_, C, B>> {
        self.open_staged()?;
        Ok(StagedBatch {
            store: StoreRef::Borrowed(self),
            run: Vec::new(),
            phase: Phase::Open,
        })
    }

    /// [`Self::staged_batch`] for a store held in an `Arc`: the batch keeps
    /// the store alive, so it can be stored beside other owners of it.
    pub fn staged_batch_shared(self: &Arc<Self>) -> Result<StagedBatch<'static, C, B>> {
        self.open_staged()?;
        Ok(StagedBatch {
            store: StoreRef::Shared(Arc::clone(self)),
            run: Vec::new(),
            phase: Phase::Open,
        })
    }

    fn open_staged(&self) -> Result<()> {
        if self.backend.conditional_append().is_none() {
            return Err(Error::ConditionalAppendUnsupported);
        }
        if !self.backend.writable() {
            return Err(Error::ReadOnly);
        }
        let mut state = self.state()?;
        self.catch_up(&mut state, None)?;
        self.checkpoint_if_due(&mut state)?;
        state.staged = true;
        if let Some(deps) = &state.deps {
            deps.set_staged(true);
        }
        Ok(())
    }

    /// Lands `run` right after `state.applied`, then advances the position
    /// and notifies subscribers record by record without applying again.
    fn commit_staged(&self, state: &mut State<C>, run: &[Record]) -> Result<Position, Failure> {
        let after = state.applied;
        if !run.is_empty() {
            let append_at = self
                .backend
                .conditional_append()
                .expect("the backend's conditional append was checked when the batch opened");
            let last = Position(after.0 + run.len() as u64);
            match append_at.append_at(after, run) {
                Ok(landed) if landed == last => {}
                Ok(_) => {
                    return Err(Failure::Poison(Error::Corrupt(
                        "backend's append_at returned a position other than the run's end",
                    )))
                }
                Err(Error::TailMoved { .. }) => self.find_run(after, run)?,
                Err(e) => return Err(Failure::Kept(e)),
            }
            if self.options.durability == Durability::Sync {
                self.backend.sync(last).map_err(Failure::Poison)?;
            }
            #[cfg(test)]
            park(StagedPark::Landed);
            for (position, record) in (after.0 + 1..).map(Position).zip(run) {
                state.applied = position;
                state.since_checkpoint += 1;
                state.notify(position, record, Origin::Local);
            }
        }
        state.staged = false;
        if let Some(deps) = &state.deps {
            deps.set_staged(false);
        }
        Ok(state.applied)
    }

    /// Checks, after `append_at` found the tail moved, that the log holds
    /// exactly `run` right after `after`: an earlier attempt that failed
    /// ambiguously landed it. Anything else there means another writer got
    /// in first.
    fn find_run(&self, after: Position, run: &[Record]) -> Result<(), Failure> {
        let mut read = Vec::new();
        match self.backend.read_from(after, &mut read) {
            Ok(ReadOutcome::Records) => {}
            Ok(ReadOutcome::Compacted { .. }) => {
                return Err(Failure::Poison(Error::TailMoved { after }))
            }
            Err(e) => return Err(Failure::Kept(e)),
        }
        let ours = read.len() >= run.len()
            && (after.0 + 1..)
                .map(Position)
                .zip(run)
                .zip(&read)
                .all(|((position, mine), (at, found))| *at == position && found == mine);
        if ours {
            Ok(())
        } else {
            Err(Failure::Poison(Error::TailMoved { after }))
        }
    }
}

impl<C: Cells, B: Backend> StagedBatch<'_, C, B> {
    /// Writes `value` to `input`, as [`Store::set`] would.
    pub fn set<T: PersistValue>(&mut self, input: Input<T>, value: T) -> Result<()> {
        self.stage(|state, rt| set_record(state, rt, input, &value))
    }

    /// Inserts `value` into `collection`, as
    /// [`PersistedCollection::insert`] would.
    pub fn insert<T: PersistValue + Hash + Eq>(
        &mut self,
        collection: &PersistedCollection<T, C>,
        value: T,
    ) -> Result<()> {
        self.stage(|state, _| collection.insert_record(state, &value))
    }

    /// Removes one occurrence of `value` from `collection`, as
    /// [`PersistedCollection::delete`] would.
    pub fn delete<T: PersistValue + Hash + Eq>(
        &mut self,
        collection: &PersistedCollection<T, C>,
        value: T,
    ) -> Result<()> {
        self.stage(|state, _| collection.delete_record(state, &value))
    }

    /// Replaces `old` with `new` in `collection`, as
    /// [`PersistedCollection::replace`] would.
    pub fn replace<T: PersistValue + Hash + Eq>(
        &mut self,
        collection: &PersistedCollection<T, C>,
        old: Option<T>,
        new: T,
    ) -> Result<()> {
        self.stage(|state, _| collection.replace_record(state, old.as_ref(), &new))
    }

    /// Applies everything `pending` collected, keyed writes included, as
    /// [`Store::write_pending`] would write it. A keyed stage built against
    /// rows an earlier staged write has since changed is repaired by
    /// further staged records.
    pub fn write_pending(&mut self, pending: PendingWrite<C>) -> Result<()> {
        self.require_open()?;
        let StagedBatch { store, run, .. } = self;
        let store = &**store;
        let mut state = store.state()?;
        store
            .write_pending_with(&mut state, pending, &mut |state, records| {
                for record in records {
                    store.apply_staged(state, record, run)?;
                    #[cfg(test)]
                    park(StagedPark::Applied);
                }
                Ok((state.applied, false))
            })
            .map(drop)
    }

    /// Appends the staged records in write order as one conditional run
    /// with one sync, and returns the last one's position, or the position
    /// the store had applied when nothing was staged.
    ///
    /// When the backend reports the tail moved, the log is read from the
    /// store's position: holding exactly this run there, it is adopted as
    /// landed; anything else poisons the store. Any other append error
    /// keeps the run for [`retry`](Self::retry), with the store still
    /// refusing other writes; a failure after the run landed poisons it.
    /// Fails with [`Error::StagedBatchClosed`] once a commit was attempted.
    pub fn commit(&mut self) -> Result<Position> {
        if self.phase != Phase::Open {
            return Err(Error::StagedBatchClosed);
        }
        self.land()
    }

    /// Attempts the commit again after it failed with an error that kept
    /// the run, with the same conditional append, so the run lands at most
    /// once. Before any commit it is [`commit`](Self::commit). Fails with
    /// [`Error::StagedBatchClosed`] once the run has landed.
    pub fn retry(&mut self) -> Result<Position> {
        if self.phase == Phase::Landed {
            return Err(Error::StagedBatchClosed);
        }
        self.land()
    }

    fn land(&mut self) -> Result<Position> {
        let StagedBatch { store, run, phase } = self;
        let mut state = store.state()?;
        match store.commit_staged(&mut state, run) {
            Ok(last) => {
                *phase = Phase::Landed;
                Ok(last)
            }
            Err(Failure::Kept(e)) => {
                *phase = Phase::AwaitingRetry;
                Err(e)
            }
            Err(Failure::Poison(e)) => {
                state.poison();
                *phase = Phase::AwaitingRetry;
                Err(e)
            }
        }
    }

    fn require_open(&self) -> Result<()> {
        if self.phase == Phase::Open {
            Ok(())
        } else {
            Err(Error::StagedBatchClosed)
        }
    }

    fn stage(
        &mut self,
        record: impl FnOnce(&State<C>, &Runtime<C>) -> Result<Record>,
    ) -> Result<()> {
        self.require_open()?;
        let StagedBatch { store, run, .. } = self;
        let mut state = store.state()?;
        let record = record(&state, store.runtime())?;
        store.apply_staged(&mut state, record, run)?;
        #[cfg(test)]
        park(StagedPark::Applied);
        Ok(())
    }
}

impl<C: Cells, B: Backend> Drop for StagedBatch<'_, C, B> {
    fn drop(&mut self) {
        if self.phase != Phase::Landed {
            self.store
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .poison();
        }
    }
}
