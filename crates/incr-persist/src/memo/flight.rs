//! `Memo::begin_checkpoint`: the checkpoint of `snapshot.rs` as a flight.
//! The caller inspects the dirty set a shard and a batch at a time in
//! `poll`, the worker resolves, encodes and writes once the caller seals,
//! and the caller then finishes the written entries by budget. Entries
//! inspected at different times are paired only through the stamps each
//! query dep recorded in its batch.

use std::any::Any;
use std::collections::hash_map::IntoIter;
use std::collections::{HashSet, VecDeque};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

use incr_core::persist_hooks::Deferred;
use incr_core::{ErasedIncr, Local, Runtime, Yielder};

use super::checkpoint::Pulls;
use super::dirty::{Record, SHARDS};
use super::memo_core::{CoreRef, ErasedBackend, IndexGen, Landing, MemoCells, MemoState};
use super::open::Memo;
use super::snapshot::{copy_seeds, fresh, stored_bytes, Origin, Seed, Seeds};
use super::table_core::{KeyPart, Snap};
use super::{CheckpointInput, CheckpointStats, Checkpointed, GenId, TableDesc};
use crate::backend::Backend;
use crate::flight::{Control, Flight, FlightToken, Inbox, Parcel, Verdict};
use crate::value_context::{EncodeMismatch, ValueScope};
use crate::{Error, Result};

/// Slots a prune moves, or written entries a finish applies, between two
/// looks at the clock.
const UNIT: usize = 4096;
/// What one inspect batch aims to take.
const BATCH_TARGET: Duration = Duration::from_millis(5);
const FIRST_BATCH: usize = 256;

/// A memo checkpoint begun with [`Memo::begin_checkpoint`]. It holds only
/// the flight's token, so it may be sent and dropped on any thread;
/// dropping it abandons the flight, which the memo then settles at its next
/// checkpoint entry point once the worker has stopped.
pub struct MemoCheckpoint {
    token: FlightToken,
}

impl MemoCheckpoint {
    /// Runs the caller's part of the checkpoint for about `budget`, plus
    /// one inspect batch or [`UNIT`] entries: the seed prune, inspecting
    /// the dirty set, and once the worker has written, the finish. `Ok(true)`
    /// once the checkpoint is finished; the worker's error, or
    /// [`Error::Cancelled`], once it is not going to be. Inspection waits
    /// while an unlogged scope or a staged batch is open.
    pub fn poll<B: Backend>(&self, memo: &Memo<Local, B>, budget: Duration) -> Result<bool> {
        memo.core().poll_flight(memo.runtime(), &self.token, budget)
    }

    /// Polls until the checkpoint is finished and returns its stats, or
    /// `Ok(None)` once `timeout` passes with the flight still open.
    /// [`Error::UnloggedScopeOpen`] or [`Error::StagedBatchOpen`] while one
    /// is open with inspection left, since only this thread can close it.
    pub fn result<B: Backend>(
        &self,
        memo: &Memo<Local, B>,
        timeout: Option<Duration>,
    ) -> Result<Option<CheckpointStats>> {
        memo.core()
            .flight_result(memo.runtime(), &self.token, timeout)
    }

    /// Stops the flight: its next poll, or the next checkpoint call, ends
    /// it and the poll returns [`Error::Cancelled`]. A flight the worker
    /// already committed ends without its finish, keeping the generation
    /// with its entries still dirty; a poll once the finish began completes
    /// the finish instead. Takes no lock.
    pub fn cancel(&self) {
        self.token.cancel();
    }
}

/// A memo's checkpoint slot: the open flight, and how the last one ended
/// until its handle reads it.
#[derive(Default)]
pub(crate) struct Slot {
    pub(crate) open: Option<MemoFlight>,
    ended: Option<Ended>,
}

struct Ended {
    token: Arc<Control>,
    outcome: Outcome,
}

enum Outcome {
    Done(CheckpointStats),
    Failed(Error),
    Panicked(Box<dyn Any + Send>),
    Reported,
}

/// What the caller sends the worker.
enum ToWorker {
    Snaps(Parcel<Snap>),
    Seeds(Vec<(ErasedIncr, Seed)>),
    /// Inspection is over: the index whose layers the write may reuse and
    /// whose base backs the stored-bytes lookups.
    Seal(Arc<IndexGen>),
}

/// What the worker needs besides its messages: no part of the memo whose
/// last drop could land on the worker.
struct Job {
    backend: Arc<dyn ErasedBackend>,
    build: u128,
    catalog: Vec<TableDesc>,
    max_bytes: u64,
    scope: Option<ValueScope>,
}

/// What the worker hands back.
enum Written {
    Nothing {
        mismatches: Vec<EncodeMismatch>,
        unsaveable: u64,
    },
    Generation(Box<Generation>),
}

struct Generation {
    done: Checkpointed,
    index: Option<IndexGen>,
    origin: Vec<Origin>,
    unsaveable: u64,
    mismatches: Vec<EncodeMismatch>,
}

enum Stage {
    Prune(Prune),
    Inspect {
        shard: usize,
        pending: VecDeque<Record>,
    },
    Sealed,
    Finish(Finishing),
}

/// The seed maps' entries a prune has yet to move into the fresh maps.
struct Prune {
    info: IntoIter<ErasedIncr, (u128, u64)>,
    keys: IntoIter<ErasedIncr, (u32, Vec<u8>)>,
}

impl Prune {
    /// Moves the rest back, alive or not, under entries written since.
    fn put_back(self, state: &mut MemoState) {
        for (handle, info) in self.info {
            state.seed_info.entry(handle).or_insert(info);
        }
        for (handle, key) in self.keys {
            state.seed_keys.entry(handle).or_insert(key);
        }
    }
}

struct Finishing {
    generation: Box<Generation>,
    next: usize,
    evicted_next: usize,
    unsaveable: u64,
}

pub(crate) struct MemoFlight {
    flight: Flight<ToWorker, Written>,
    stage: Stage,
    started: Instant,
    batch: usize,
    copied: HashSet<ErasedIncr>,
    /// What the inspect batches carried, handed over as one checkpoint's
    /// landing at the seal: a later handover replaces an earlier one.
    landing: Landing,
}

impl MemoFlight {
    fn is(&self, token: &FlightToken) -> bool {
        token.is_of(self.flight.control())
    }

    /// Whether an entry point other than `poll` settles this flight now:
    /// its worker has stopped and nothing is left that only `poll` runs. A
    /// committed write on a live handle still owes its finish to `poll`.
    fn settles_idle(&self) -> bool {
        let control = self.flight.control();
        match self.stage {
            Stage::Finish(_) => control.stopped(),
            _ => {
                self.flight.has_verdict() && (control.stopped() || !self.flight.posted_committed())
            }
        }
    }
}

impl<C: MemoCells> CoreRef<C> {
    /// Settles an open flight an entry point may settle; returns whether
    /// none is open after it.
    pub(crate) fn settle_idle(&self, slot: &mut Slot) -> bool {
        match &slot.open {
            None => true,
            Some(flight) if flight.settles_idle() => {
                let flight = slot.open.take().expect("checked open");
                slot.ended = Some(self.settle(flight, None));
                true
            }
            Some(_) => false,
        }
    }

    /// Whether a flight is open after settling what an entry point may.
    pub(crate) fn flight_open(&self) -> bool {
        !self.settle_idle(&mut self.lock_checkpoint())
    }

    /// Ends `flight` without its finish: puts back the seeds its prune
    /// had not moved, swaps in the index of a generation the worker
    /// committed and removes the generations it superseded, and takes the
    /// attempt back so its entries stay dirty. `verdict` is the worker's,
    /// if already taken. Returns what the handle is to read.
    fn settle(&self, mut flight: MemoFlight, verdict: Option<Verdict<Written>>) -> Ended {
        let verdict = verdict.or_else(|| flight.flight.take_verdict());
        let mut state = self.lock_memo();
        if let Stage::Prune(prune) = std::mem::replace(&mut flight.stage, Stage::Sealed) {
            prune.put_back(&mut state);
        }
        let outcome = match verdict {
            Some(Verdict::Committed(Written::Generation(mut generation))) => {
                self.publish(&mut state, &mut generation);
                Outcome::Failed(Error::Cancelled)
            }
            Some(Verdict::Committed(Written::Nothing { mismatches, .. })) => {
                for mismatch in mismatches {
                    state.record_mismatch(mismatch);
                }
                Outcome::Failed(Error::Cancelled)
            }
            Some(Verdict::Failed(error)) => Outcome::Failed(error),
            Some(Verdict::Truncated) => Outcome::Failed(Error::TruncatedWrite),
            Some(Verdict::Panicked(payload)) => Outcome::Panicked(payload),
            Some(Verdict::Cancelled) | None => Outcome::Failed(Error::Cancelled),
        };
        drop(state);
        self.dirty.failed();
        let token = Arc::clone(flight.flight.control());
        drop(flight);
        Ended { token, outcome }
    }

    /// Swaps in a committed generation's index, counts the write and
    /// starts the removal of the generations it superseded; the caller
    /// holds the memo lock.
    fn publish(&self, state: &mut MemoState, generation: &mut Generation) {
        let Some(index) = generation.index.take() else {
            return;
        };
        for mismatch in std::mem::take(&mut generation.mismatches) {
            state.record_mismatch(mismatch);
        }
        self.swap_index(index);
        state.count_written(&generation.done.stats);
        remove_in_background(
            Arc::clone(&self.backend),
            std::mem::take(&mut generation.done.superseded),
            Arc::clone(&self.stale_generations),
            self.scope.clone(),
        );
    }

    /// The caller side of [`MemoCheckpoint::poll`].
    pub(crate) fn poll_flight(
        &self,
        rt: &Runtime<C>,
        token: &FlightToken,
        budget: Duration,
    ) -> Result<bool> {
        self.check_call(rt)?;
        let deadline = Instant::now() + budget;
        let mut slot = self.lock_checkpoint();
        if !slot.open.as_ref().is_some_and(|flight| flight.is(token)) {
            return take_ended(&mut slot, token);
        }
        let stepped = panic::catch_unwind(AssertUnwindSafe(|| {
            self.drive(rt, slot.open.as_mut().expect("checked open"), deadline)
        }));
        let ended = match stepped {
            Ok(Step::Open) => return Ok(false),
            Ok(Step::Finished(stats)) => {
                let flight = slot.open.take().expect("checked open");
                slot.ended = Some(Ended {
                    token: Arc::clone(flight.flight.control()),
                    outcome: Outcome::Done(stats),
                });
                return Ok(true);
            }
            Ok(Step::Ended(verdict)) => {
                let flight = slot.open.take().expect("checked open");
                self.settle(flight, Some(verdict))
            }
            Ok(Step::Failed(error)) => {
                let flight = slot.open.take().expect("checked open");
                flight.flight.cancel();
                let mut ended = self.settle(flight, Some(Verdict::Cancelled));
                ended.outcome = Outcome::Failed(error);
                ended
            }
            Err(payload) => {
                let flight = slot.open.take().expect("checked open");
                flight.flight.cancel();
                let mut ended = self.settle(flight, Some(Verdict::Cancelled));
                ended.outcome = Outcome::Panicked(payload);
                ended
            }
        };
        slot.ended = Some(ended);
        take_ended(&mut slot, token)
    }

    /// The caller side of [`MemoCheckpoint::result`].
    pub(crate) fn flight_result(
        &self,
        rt: &Runtime<C>,
        token: &FlightToken,
        timeout: Option<Duration>,
    ) -> Result<Option<CheckpointStats>> {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        loop {
            if self.poll_flight(rt, token, Duration::from_millis(50))? {
                let slot = self.lock_checkpoint();
                return Ok(match &slot.ended {
                    Some(Ended {
                        outcome: Outcome::Done(stats),
                        ..
                    }) => Some(stats.clone()),
                    _ => None,
                });
            }
            let slot = self.lock_checkpoint();
            let Some(flight) = slot.open.as_ref().filter(|flight| flight.is(token)) else {
                continue;
            };
            if matches!(flight.stage, Stage::Prune(_) | Stage::Inspect { .. }) {
                if self.deps.unlogged() {
                    return Err(Error::UnloggedScopeOpen);
                }
                if self.deps.staged() {
                    return Err(Error::StagedBatchOpen);
                }
            }
            let left = match deadline {
                None => None,
                Some(deadline) => match deadline.checked_duration_since(Instant::now()) {
                    Some(left) if !left.is_zero() => Some(left),
                    _ => return Ok(None),
                },
            };
            if matches!(flight.stage, Stage::Sealed) {
                flight.flight.wait(left);
            }
        }
    }

    fn check_call(&self, rt: &Runtime<C>) -> Result<()> {
        assert!(
            !rt.frame_active(),
            "incr-persist: a memo checkpoint polled from inside a query's compute"
        );
        if incr_core::in_value_code() {
            return Err(Error::Reentrant);
        }
        Ok(())
    }

    /// Runs stages until `deadline`, the flight's end, or a stage that
    /// waits.
    fn drive(&self, rt: &Runtime<C>, flight: &mut MemoFlight, deadline: Instant) -> Step {
        loop {
            if !matches!(flight.stage, Stage::Finish(_)) && flight.flight.has_verdict() {
                let verdict = flight.flight.take_verdict().expect("has a verdict");
                match verdict {
                    Verdict::Committed(written) if !flight.flight.control().stopped() => {
                        match self.begin_finish(flight, written) {
                            Some(stats) => return Step::Finished(stats),
                            None => continue,
                        }
                    }
                    verdict => return Step::Ended(verdict),
                }
            }
            if flight.flight.control().stopped() && !matches!(flight.stage, Stage::Finish(_)) {
                if matches!(flight.stage, Stage::Sealed) {
                    return Step::Open;
                }
                return Step::Ended(Verdict::Cancelled);
            }
            if self.deps.poisoned()
                && matches!(flight.stage, Stage::Prune(_) | Stage::Inspect { .. })
            {
                return Step::Failed(Error::Poisoned);
            }
            let progressed = match flight.stage {
                Stage::Prune(_) => self.prune_slice(rt, flight),
                Stage::Inspect { .. } => {
                    if self.deps.unlogged() || self.deps.staged() {
                        return Step::Open;
                    }
                    self.inspect_slice(rt, flight)
                }
                Stage::Sealed => return Step::Open,
                Stage::Finish(_) => match self.finish_slice(rt, flight) {
                    Some(stats) => return Step::Finished(stats),
                    None => true,
                },
            };
            if !progressed || Instant::now() >= deadline {
                return Step::Open;
            }
        }
    }

    fn prune_slice(&self, rt: &Runtime<C>, flight: &mut MemoFlight) -> bool {
        let Stage::Prune(prune) = &mut flight.stage else {
            return false;
        };
        let mut state = self.lock_memo();
        let mut moved = 0;
        for (handle, info) in prune.info.by_ref() {
            if rt.is_alive(handle) {
                state.seed_info.entry(handle).or_insert(info);
            }
            moved += 1;
            if moved == UNIT {
                return true;
            }
        }
        for (handle, key) in prune.keys.by_ref() {
            if rt.is_alive(handle) {
                state.seed_keys.entry(handle).or_insert(key);
            }
            moved += 1;
            if moved == UNIT {
                return true;
            }
        }
        state.pruned_at = state.seed_info.len();
        flight.stage = Stage::Inspect {
            shard: 0,
            pending: VecDeque::new(),
        };
        true
    }

    /// Inspects one batch of one table's dirty entries, or copies the next
    /// shard's, or seals once every shard is inspected.
    fn inspect_slice(&self, rt: &Runtime<C>, flight: &mut MemoFlight) -> bool {
        let Stage::Inspect { shard, pending } = &mut flight.stage else {
            return false;
        };
        if pending.is_empty() {
            if *shard == SHARDS {
                self.wake.await_landing(std::mem::take(&mut flight.landing));
                flight.flight.send(ToWorker::Seal(self.index()));
                flight.stage = Stage::Sealed;
                return true;
            }
            let mut dead = Vec::new();
            *pending = self
                .dirty
                .take_shard(*shard, |handle| rt.is_alive(handle), &mut dead)
                .into();
            *shard += 1;
            drop(dead);
            return true;
        }
        let table = pending[0].table;
        let take = pending
            .iter()
            .take(flight.batch)
            .take_while(|record| record.table == table)
            .count();
        let batch: Vec<Record> = pending.drain(..take).collect();
        let refs: Vec<&Record> = batch.iter().collect();
        let mut bag = Deferred::<C>::new();
        let started = Instant::now();
        let (snaps, seeds) = {
            let state = self.lock_memo();
            let yielder = rt.yielder_under_lock();
            let map = self.deps.read();
            let mut snaps = Vec::new();
            self.tables[table as usize]
                .inspect_dirty(rt, &map, self.id, &refs, &mut snaps, &mut bag);
            drop(map);
            let seeds = copy_seeds(&state, &mut snaps, &mut flight.copied, &yielder);
            (snaps, seeds)
        };
        let took = started.elapsed();
        self.carry_uninspected(&batch, &snaps, &Yielder::ambient(), &mut flight.landing);
        flight.batch = next_batch(flight.batch, take, took);
        if !snaps.is_empty() {
            flight
                .flight
                .send(ToWorker::Snaps(Parcel::new(snaps, self.scope.clone())));
        }
        if !seeds.is_empty() {
            flight.flight.send(ToWorker::Seeds(seeds));
        }
        drop(refs);
        drop(batch);
        bag.drain(rt);
        true
    }

    /// Starts the finish of what the worker wrote; the stats if nothing is
    /// left to finish.
    fn begin_finish(&self, flight: &mut MemoFlight, written: Written) -> Option<CheckpointStats> {
        let snapshot = flight.started.elapsed();
        match written {
            Written::Nothing {
                mismatches,
                unsaveable,
            } => Some(self.unchanged(mismatches, unsaveable, snapshot)),
            Written::Generation(mut generation) => {
                generation.done.stats.times.snapshot = snapshot;
                self.publish(&mut self.lock_memo(), &mut generation);
                let unsaveable = generation.unsaveable;
                flight.stage = Stage::Finish(Finishing {
                    generation,
                    next: 0,
                    evicted_next: 0,
                    unsaveable,
                });
                None
            }
        }
    }

    /// Applies up to [`UNIT`] written entries' fates, then the evicted
    /// base entries; the stats once all are applied.
    fn finish_slice(&self, rt: &Runtime<C>, flight: &mut MemoFlight) -> Option<CheckpointStats> {
        let Stage::Finish(finishing) = &mut flight.stage else {
            return None;
        };
        let done = &finishing.generation.done;
        let origin = &finishing.generation.origin;
        if finishing.next < done.fates.len() {
            let end = (finishing.next + UNIT).min(done.fates.len());
            let mut saved: Vec<KeyPart> = Vec::new();
            let mut state = self.lock_memo();
            for (fate, origin) in done.fates[finishing.next..end]
                .iter()
                .zip(&origin[finishing.next..end])
            {
                finishing.unsaveable +=
                    u64::from(self.finish_entry(&mut state, *fate, origin, &mut saved));
            }
            drop(state);
            drop(saved);
            finishing.next = end;
            return None;
        }
        if finishing.evicted_next < done.evicted_base.len() {
            let end = (finishing.evicted_next + UNIT).min(done.evicted_base.len());
            let mut bag = Deferred::<C>::new();
            for (table, key) in &done.evicted_base[finishing.evicted_next..end] {
                self.probe_evicted(rt, *table, key, &mut bag);
            }
            bag.drain(rt);
            finishing.evicted_next = end;
            return None;
        }
        let mut state = self.lock_memo();
        state.stats.unsaveable = finishing.unsaveable;
        state.stats.evicted_last_checkpoint = done.stats.evicted as u64;
        Some(done.stats.clone())
    }
}

enum Step {
    Open,
    Finished(CheckpointStats),
    Ended(Verdict<Written>),
    Failed(Error),
}

fn take_ended(slot: &mut Slot, token: &FlightToken) -> Result<bool> {
    let Some(ended) = slot
        .ended
        .as_mut()
        .filter(|ended| token.is_of(&ended.token))
    else {
        return Err(Error::UnknownHandle);
    };
    match std::mem::replace(&mut ended.outcome, Outcome::Reported) {
        Outcome::Done(stats) => {
            ended.outcome = Outcome::Done(stats);
            Ok(true)
        }
        Outcome::Failed(error) => Err(error),
        Outcome::Panicked(payload) => panic::resume_unwind(payload),
        Outcome::Reported => Err(Error::UnknownHandle),
    }
}

/// The next inspect batch size, aiming at [`BATCH_TARGET`] from what `took`
/// cost for `taken` entries.
fn next_batch(batch: usize, taken: usize, took: Duration) -> usize {
    if taken < batch {
        return batch;
    }
    let per = took.as_nanos().max(1) / taken as u128;
    let fits = (BATCH_TARGET.as_nanos() / per.max(1)) as usize;
    fits.clamp(16, UNIT).min(batch * 2).max(batch / 2)
}

impl<B: Backend> Memo<Local, B> {
    /// Starts a checkpoint that runs as a flight: each
    /// [`MemoCheckpoint::poll`] inspects the dirty set for about its
    /// budget, and a worker thread encodes and writes once inspection is
    /// over, with no memo lock held. Refuses what [`Self::checkpoint`]
    /// refuses, and a backend that does not declare
    /// [`MemoBackend::concurrent_compaction`](super::MemoBackend::concurrent_compaction).
    /// While the flight is open, `checkpoint` and another
    /// `begin_checkpoint` return [`Error::Busy`] and
    /// [`Self::checkpoint_if_due`] skips.
    pub fn begin_checkpoint(&self) -> Result<MemoCheckpoint> {
        self.core().begin_flight(self.runtime())
    }
}

impl CoreRef<Local> {
    fn begin_flight(&self, rt: &Runtime<Local>) -> Result<MemoCheckpoint> {
        self.check_checkpoint(rt)?;
        if !self.backend.concurrent_compaction() {
            return Err(Error::ConcurrentCompactionUnsupported);
        }
        let mut slot = self.lock_checkpoint();
        if !self.settle_idle(&mut slot) {
            return Err(Error::Busy);
        }
        let job = Job {
            backend: Arc::clone(&self.backend),
            build: self.build,
            catalog: self.catalog.clone(),
            max_bytes: self.max_bytes,
            scope: self.scope.clone(),
        };
        self.dirty.begin();
        let (flight, token) = match Flight::start(self.scope.clone(), move |inbox| work(inbox, job))
        {
            Ok(started) => started,
            Err(error) => {
                self.dirty.failed();
                return Err(error);
            }
        };
        let stage = {
            let mut state = self.lock_memo();
            if state.prune_due() {
                Stage::Prune(Prune {
                    info: std::mem::take(&mut state.seed_info).into_iter(),
                    keys: std::mem::take(&mut state.seed_keys).into_iter(),
                })
            } else {
                Stage::Inspect {
                    shard: 0,
                    pending: VecDeque::new(),
                }
            }
        };
        slot.open = Some(MemoFlight {
            flight,
            stage,
            started: Instant::now(),
            batch: FIRST_BATCH,
            copied: HashSet::new(),
            landing: Landing::default(),
        });
        slot.ended = None;
        Ok(MemoCheckpoint { token })
    }
}

/// The worker: gathers the snaps and seeds until the seal, then resolves
/// and encodes them, writes the generation and builds its index.
fn work(inbox: Inbox<ToWorker>, job: Job) -> Verdict<Written> {
    let mut snaps = Parcel::new(Vec::new(), job.scope.clone());
    let mut seeds = Seeds::new();
    let index = loop {
        match inbox.recv() {
            None => return Verdict::Cancelled,
            Some(ToWorker::Snaps(batch)) => snaps.append(batch),
            Some(ToWorker::Seeds(batch)) => {
                for (handle, seed) in batch {
                    seeds.entry(handle).or_insert(seed);
                }
            }
            Some(ToWorker::Seal(index)) => break index,
        }
    };
    snaps
        .as_mut_slice()
        .sort_unstable_by_key(|snap| (snap.table, snap.mark));
    let mut mismatches = Vec::new();
    let backend = &*job.backend;
    let stored = |table: u32, key: &[u8], fp: u128| stored_bytes(&index, backend, table, key, fp);
    let found = fresh(snaps, &seeds, &stored, &mut mismatches, &Yielder::ambient());
    drop(seeds);
    if found.entries.is_empty() {
        return Verdict::Committed(Written::Nothing {
            mismatches,
            unsaveable: found.unsaveable,
        });
    }
    if inbox.control().stopped() {
        return Verdict::Cancelled;
    }
    let pulls = Pulls {
        control: Arc::clone(inbox.control()),
        scope: job.scope.clone(),
    };
    let known = index.layers.clone();
    let written = job.backend.checkpoint(
        CheckpointInput {
            build: job.build,
            tables: job.catalog.clone(),
            fresh: found.entries,
            max_bytes: job.max_bytes,
        },
        &known,
        Some(&pulls),
    );
    drop(known);
    drop(index);
    match written {
        Err(Error::Cancelled) => Verdict::Cancelled,
        Err(Error::TruncatedWrite) => Verdict::Truncated,
        Err(error) => Verdict::Failed(error),
        Ok(done) => {
            let index = IndexGen::new(done.chain.clone(), job.build, &job.catalog);
            Verdict::Committed(Written::Generation(Box::new(Generation {
                done,
                index: Some(index),
                origin: found.origin,
                unsaveable: found.unsaveable,
                mismatches,
            })))
        }
    }
}

/// Removes `superseded` on a worker of its own, counting what it fails to
/// remove into `stale`. When no thread starts they are only counted: the
/// caller holds the memo lock, and the next `lock_checkpoint` drops every
/// generation outside the chain.
fn remove_in_background(
    backend: Arc<dyn ErasedBackend>,
    superseded: Vec<GenId>,
    stale: Arc<AtomicU64>,
    scope: Option<ValueScope>,
) {
    if superseded.is_empty() {
        return;
    }
    let count = superseded.len() as u64;
    let remove = {
        let stale = Arc::clone(&stale);
        move |_: Inbox<()>| {
            let left = backend.remove_generations(&superseded);
            stale.fetch_add(left.len() as u64, Relaxed);
            Verdict::Committed(())
        }
    };
    if Flight::<(), ()>::start(scope, remove).is_err() {
        stale.fetch_add(count, Relaxed);
    }
}

#[cfg(test)]
#[path = "flight_tests.rs"]
mod tests;
