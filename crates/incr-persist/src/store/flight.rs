//! `Store::begin_checkpoint`: the checkpoint of `checkpoint_locked` as a
//! flight. Begin fixes the position, shares each key's logged entry and
//! installs a capture on every bound collection; `poll` hands the captured
//! rows to a worker table by table, and the worker folds them with the
//! shared entries and compacts the backend, holding nothing of the store.
//! The store settles the worker's verdict at its next checkpoint entry point.

use std::any::Any;
use std::collections::BTreeSet;
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::time::{Duration, Instant};

use incr_core::{Cells, Local, Yielder};

use super::after_lock::AfterLock;
use super::binding::Folded;
use super::logged::LoggedKey;
use super::rows::RowBatch;
use super::{fold_logged, Locked, State, Store, TRUNCATED};
use crate::flight::{Chunked, Control, Flight, FlightToken, Inbox, Verdict, Waiter};
use crate::{Backend, Error, Position, Record, Result, SnapshotId, ValueScope};

/// How long [`StoreCheckpoint::result`] polls before it looks at the clock
/// again, and how much an `undeclare` captures between two sends.
const SLICE: Duration = Duration::from_millis(50);

/// A store checkpoint begun with [`Store::begin_checkpoint`]. It holds only
/// the flight's token, so it may be sent and dropped on any thread;
/// dropping it abandons the flight, which the store then settles at its
/// next checkpoint entry point, a plain write included, once the worker has
/// posted its verdict.
pub struct StoreCheckpoint {
    token: FlightToken,
    upto: Position,
}

impl StoreCheckpoint {
    /// Hands the worker what the captures take for about `budget`, plus
    /// one unit of table slots, and settles the flight once the worker's
    /// verdict is in. `Ok(true)` once the snapshot is committed and
    /// adopted; the worker's error or [`Error::Cancelled`] once it is not
    /// going to be, and [`Error::Corrupt`], poisoning the store, when the
    /// backend returned before it pulled the whole snapshot.
    pub fn poll<B: Backend + 'static>(
        &self,
        store: &Store<Local, B>,
        budget: Duration,
    ) -> Result<bool> {
        store.poll_flight(&self.token, budget)
    }

    /// Polls until the flight ends and returns the position its snapshot
    /// covers, or `Ok(None)` once `timeout` passes with the flight still
    /// open. Waits for the worker without the store's lock.
    pub fn result<B: Backend + 'static>(
        &self,
        store: &Store<Local, B>,
        timeout: Option<Duration>,
    ) -> Result<Option<Position>> {
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        loop {
            if store.poll_flight(&self.token, SLICE)? {
                return Ok(Some(self.upto));
            }
            let left = match deadline {
                None => None,
                Some(deadline) => match deadline.checked_duration_since(Instant::now()) {
                    Some(left) if !left.is_zero() => Some(left),
                    _ => return Ok(None),
                },
            };
            if let Some(waiter) = store.idle_flight_waiter(&self.token)? {
                waiter.wait(left);
            }
        }
    }

    /// Stops a flight whose backend has not committed; its next poll, or
    /// the store's next checkpoint entry point once the worker stopped,
    /// ends it with [`Error::Cancelled`]. One that already committed is
    /// adopted. Takes no lock.
    pub fn cancel(&self) {
        self.token.cancel();
    }
}

/// A store's checkpoint flight, if one is open, and how the last one ended
/// until its handle reads it.
#[derive(Default)]
pub(super) struct Slot {
    open: Option<StoreFlight>,
    ended: Option<Ended>,
}

impl Slot {
    /// Counts a record applied, wherever `since_checkpoint` counts one.
    pub(super) fn count_applied(&mut self) {
        if let Some(flight) = &mut self.open {
            flight.applied += 1;
        }
    }
}

struct Ended {
    control: Arc<Control>,
    outcome: Outcome,
}

enum Outcome {
    Done,
    Failed(Error),
    Panicked(Box<dyn Any + Send>),
    Reported,
}

struct StoreFlight {
    flight: Flight<ToWorker, Vec<usize>>,
    upto: Position,
    own: SnapshotId,
    /// Records applied since begin.
    applied: u64,
    /// Every bound collection's capture, in key order.
    tables: Vec<Table>,
    /// The first entry of `tables` whose capture is not finished.
    next: usize,
    sealed: bool,
    /// The keys whose fold was empty at begin, in key order, with their
    /// entries as of begin.
    forgotten: Vec<Forgotten>,
    scope: Option<ValueScope>,
}

struct Table {
    job: usize,
    binding: u32,
    done: bool,
}

struct Forgotten {
    job: usize,
    key: String,
    logged: Arc<LoggedKey>,
}

enum ToWorker {
    Rows(usize, Box<dyn RowBatch>),
    Seal,
}

/// A key the worker folds: its entry as of begin, and for a bound
/// collection the rows its capture hands over.
struct Job {
    key: String,
    logged: Option<Arc<LoggedKey>>,
    rows: Option<Box<dyn RowBatch>>,
}

impl<C: Cells> State<C> {
    /// Ends an open flight without adopting it, whatever its worker still
    /// posts, and removes every capture it installed. `outcome` is what its
    /// handle reads.
    fn abort_flight(&mut self, outcome: Outcome) {
        let Some(flight) = self.flight.open.take() else {
            return;
        };
        flight.flight.cancel();
        self.end_captures(&flight);
        self.set_ended(Some(Ended {
            control: Arc::clone(flight.flight.control()),
            outcome,
        }));
    }

    /// Replaces how the last flight ended; a panic payload the replaced
    /// one still holds drops after the lock.
    fn set_ended(&mut self, ended: Option<Ended>) {
        if let Some(replaced) = mem::replace(&mut self.flight.ended, ended) {
            self.after.push(replaced);
        }
    }

    /// [`Self::abort_flight`] for a store being poisoned.
    pub(super) fn abort_flight_poisoned(&mut self) {
        self.abort_flight(Outcome::Failed(Error::Poisoned));
    }

    /// [`Self::abort_flight`] for a rebuild of the folded log.
    pub(super) fn abort_flight_rebuilt(&mut self) {
        self.abort_flight(Outcome::Failed(Error::Cancelled));
    }

    fn end_captures(&self, flight: &StoreFlight) {
        for table in &flight.tables {
            if let Some(binding) = self.bindings.get(&table.binding) {
                binding.end_capture(&self.after);
            }
        }
    }

    /// Runs the capture of binding `id` to its end and hands the rows to
    /// the worker, so the snapshot keeps the collection as of begin after
    /// the binding goes.
    pub(super) fn finish_capture(&mut self, id: u32) {
        let State {
            flight,
            bindings,
            after,
            ..
        } = self;
        let Some(flight) = &mut flight.open else {
            return;
        };
        let Some(table) = flight
            .tables
            .iter_mut()
            .find(|table| table.binding == id && !table.done)
        else {
            return;
        };
        let binding = &bindings[&id];
        loop {
            let (rows, done) = binding.capture_step(SLICE, flight.scope.clone(), after);
            if let Some(rows) = rows {
                send(&flight.flight, ToWorker::Rows(table.job, rows), after);
            }
            if done {
                break;
            }
        }
        table.done = true;
    }
}

impl<C: Cells, B: Backend> Store<C, B> {
    /// Settles an open flight whose worker has posted its verdict and
    /// returns whether a flight is still open. A snapshot the backend may
    /// hold truncated poisons the store and fails with [`Error::Corrupt`].
    pub(super) fn settle_flight(&self, state: &mut State<C>) -> Result<bool> {
        let Some(open) = &state.flight.open else {
            return Ok(false);
        };
        if !open.flight.has_verdict() {
            return Ok(true);
        }
        let flight = state.flight.open.take().expect("checked open");
        state.end_captures(&flight);
        let outcome = match flight.flight.take_verdict().expect("has a verdict") {
            Verdict::Committed(empty) => {
                self.adopt(state, &flight, &empty);
                Outcome::Done
            }
            Verdict::Truncated => {
                state.poison();
                state.set_ended(Some(Ended {
                    control: Arc::clone(flight.flight.control()),
                    outcome: Outcome::Reported,
                }));
                return Err(Error::Corrupt(TRUNCATED));
            }
            Verdict::Failed(error) => Outcome::Failed(error),
            Verdict::Cancelled => Outcome::Failed(Error::Cancelled),
            Verdict::Panicked(payload) => Outcome::Panicked(payload),
        };
        state.set_ended(Some(Ended {
            control: Arc::clone(flight.flight.control()),
            outcome,
        }));
        Ok(false)
    }

    /// Takes the committed snapshot as this store's checkpoint at the
    /// flight's position. A key forgotten at begin is dropped only while
    /// its entry is the one begin shared; when any changed, the snapshot
    /// stays unaccounted and the next reconcile rebuilds from it.
    fn adopt(&self, state: &mut State<C>, flight: &StoreFlight, empty: &[usize]) {
        debug_assert!(
            flight
                .forgotten
                .iter()
                .map(|forgotten| forgotten.job)
                .eq(empty.iter().copied()),
            "the keys a flight's worker folded to nothing are those its begin forgot"
        );
        let mut unchanged = true;
        for forgotten in &flight.forgotten {
            match state.logged.shared(&forgotten.key) {
                Some(entry) if Arc::ptr_eq(&entry, &forgotten.logged) => {
                    drop(entry);
                    state.logged.remove(&forgotten.key);
                }
                _ => unchanged = false,
            }
        }
        state.since_checkpoint = flight.applied;
        state.snapshot_at = flight.upto;
        if unchanged && matches!(self.backend.snapshot_id(), Ok(Some(id)) if id == flight.own) {
            state.snapshot = Some(flight.own);
        }
    }

    /// The open flight's verdict to wait on, when `token` is its handle and
    /// only the worker has work left.
    fn idle_flight_waiter(&self, token: &FlightToken) -> Result<Option<Waiter<Vec<usize>>>> {
        let state = self.state()?;
        Ok(state
            .flight
            .open
            .as_ref()
            .filter(|flight| {
                token.is_of(flight.flight.control())
                    && (flight.sealed || flight.flight.control().stopped())
            })
            .map(|flight| flight.flight.waiter()))
    }

    /// Waits for an open flight's verdict without settling it; whether
    /// none is open or its verdict is in.
    #[cfg(test)]
    pub(crate) fn wait_flight_verdict(&self, timeout: Duration) -> bool {
        let waiter = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state
                .flight
                .open
                .as_ref()
                .map(|flight| flight.flight.waiter())
        };
        waiter.is_none_or(|waiter| waiter.wait(Some(timeout)))
    }
}

impl<B: Backend + 'static> Store<Local, B> {
    /// Starts a checkpoint that runs as a flight: it fixes the position as
    /// [`Self::checkpoint`] would, each [`StoreCheckpoint::poll`] copies the
    /// bound collections for about its budget, and a worker thread folds,
    /// encodes and compacts while writes go on. Refuses what `checkpoint`
    /// refuses, and a backend that does not declare
    /// [`Backend::concurrent_compaction`]. While the flight is open and its
    /// worker has posted no verdict, `checkpoint` and another
    /// `begin_checkpoint` return [`Error::Busy`] and the automatic
    /// checkpoint is skipped.
    pub fn begin_checkpoint(&self) -> Result<StoreCheckpoint> {
        let mut guard = self.state()?;
        let state = &mut *guard;
        if self.settle_flight(state)? {
            return Err(Error::Busy);
        }
        self.catch_up(state, None)?;
        #[cfg(debug_assertions)]
        for (key, id) in &state.bound {
            state.bindings[id].check_bypass(&self.runtime, state.logged.get(key), &state.after)?;
        }
        if !self.backend.writable() {
            return Err(Error::ReadOnly);
        }
        if !self.backend.concurrent_compaction() {
            return Err(Error::ConcurrentCompactionUnsupported);
        }
        let upto = state.applied;
        let own = SnapshotId {
            position: upto,
            generation: state.snapshot.map_or(0, |id| id.generation) + 1,
        };
        let scope = self.value_scope();
        let keys: BTreeSet<&String> = state.logged.keys().chain(state.bound.keys()).collect();
        let mut jobs = Vec::with_capacity(keys.len());
        let mut tables = Vec::new();
        let mut forgotten = Vec::new();
        for key in keys {
            let logged = state.logged.shared(key);
            let (empty, rows) = match state.bound.get(key) {
                Some(&id) => {
                    let binding = &state.bindings[&id];
                    let empty = binding.folds_empty(logged.as_deref());
                    let rows = binding.begin_capture(scope.clone());
                    if rows.is_some() {
                        tables.push(Table {
                            job: jobs.len(),
                            binding: id,
                            done: false,
                        });
                    }
                    (empty, rows)
                }
                None => (logged.as_deref().is_none_or(LoggedKey::at_default), None),
            };
            if let Some(logged) = logged.as_ref().filter(|_| empty) {
                forgotten.push(Forgotten {
                    job: jobs.len(),
                    key: key.clone(),
                    logged: Arc::clone(logged),
                });
            }
            jobs.push(Job {
                key: key.clone(),
                logged,
                rows,
            });
        }
        let backend = Arc::clone(&self.backend);
        let worker_scope = scope.clone();
        let started = Flight::start(scope.clone(), move |inbox| {
            work(inbox, jobs, &*backend, upto, worker_scope)
        });
        let (flight, token) = match started {
            Ok(started) => started,
            Err(error) => {
                for table in &tables {
                    state.bindings[&table.binding].end_capture(&state.after);
                }
                return Err(error);
            }
        };
        state.flight.open = Some(StoreFlight {
            flight,
            upto,
            own,
            applied: 0,
            tables,
            next: 0,
            sealed: false,
            forgotten,
            scope,
        });
        state.set_ended(None);
        Ok(StoreCheckpoint { token, upto })
    }

    /// The caller side of [`StoreCheckpoint::poll`].
    fn poll_flight(&self, token: &FlightToken, budget: Duration) -> Result<bool> {
        let deadline = Instant::now() + budget;
        let mut guard = self.state()?;
        let state = &mut *guard;
        let ours = state
            .flight
            .open
            .as_ref()
            .is_some_and(|flight| token.is_of(flight.flight.control()));
        if !ours || !self.settle_flight(state)? {
            return take_ended(guard, token);
        }
        let stepped = panic::catch_unwind(AssertUnwindSafe(|| capture_slice(state, deadline)));
        if let Err(payload) = stepped {
            state.abort_flight(Outcome::Reported);
            drop(guard);
            panic::resume_unwind(payload);
        }
        Ok(false)
    }
}

/// Runs the open flight's captures in key order until `deadline`, at least
/// one unit of table slots, sending each run to the worker, and the seal
/// once every table is done.
fn capture_slice(state: &mut State<Local>, deadline: Instant) {
    let State {
        flight,
        bindings,
        after,
        ..
    } = state;
    let flight = flight.open.as_mut().expect("an open flight");
    if flight.flight.control().stopped() {
        return;
    }
    while !flight.sealed {
        let Some(table) = flight.tables.get_mut(flight.next) else {
            send(&flight.flight, ToWorker::Seal, after);
            flight.sealed = true;
            return;
        };
        if let Some(binding) = bindings.get(&table.binding).filter(|_| !table.done) {
            let left = deadline.saturating_duration_since(Instant::now());
            let (rows, done) = binding.capture_step(left, flight.scope.clone(), after);
            if let Some(rows) = rows {
                send(&flight.flight, ToWorker::Rows(table.job, rows), after);
            }
            table.done = done;
        } else {
            table.done = true;
        }
        if table.done {
            flight.next += 1;
        }
        if Instant::now() >= deadline {
            return;
        }
    }
}

/// Sends `message` to `flight`'s worker; one the stopped worker no longer
/// takes drops after the lock.
fn send(flight: &Flight<ToWorker, Vec<usize>>, message: ToWorker, after: &AfterLock) {
    if let Some(unsent) = flight.send(message) {
        after.push(unsent);
    }
}

/// What the handle of a flight that is no longer open reads: its outcome
/// once, `Ok(true)` for as long as it ended adopted.
fn take_ended<C: Cells>(mut guard: Locked<'_, C>, token: &FlightToken) -> Result<bool> {
    let Some(ended) = guard
        .flight
        .ended
        .as_mut()
        .filter(|ended| token.is_of(&ended.control))
    else {
        return Err(Error::UnknownHandle);
    };
    match mem::replace(&mut ended.outcome, Outcome::Reported) {
        Outcome::Done => {
            ended.outcome = Outcome::Done;
            Ok(true)
        }
        Outcome::Failed(error) => Err(error),
        Outcome::Panicked(payload) => {
            drop(guard);
            panic::resume_unwind(payload)
        }
        Outcome::Reported => Err(Error::UnknownHandle),
    }
}

/// The worker: gathers the captured rows until the seal, then folds every
/// key in order and hands the records to the backend through [`Chunked`].
/// Commits with the jobs whose key folded to nothing.
fn work<B: Backend>(
    inbox: Inbox<ToWorker>,
    mut jobs: Vec<Job>,
    backend: &B,
    upto: Position,
    scope: Option<ValueScope>,
) -> Verdict<Vec<usize>> {
    loop {
        match inbox.recv() {
            None => return Verdict::Cancelled,
            Some(ToWorker::Rows(job, rows)) => jobs[job]
                .rows
                .as_mut()
                .expect("rows only for a captured collection")
                .absorb(rows),
            Some(ToWorker::Seal) => break,
        }
    }
    if inbox.control().stopped() {
        return Verdict::Cancelled;
    }
    let (keys, rows): (Vec<_>, Vec<_>) = jobs
        .into_iter()
        .map(|job| ((job.key, job.logged), job.rows))
        .unzip();
    let fold = FlightFold {
        keys: &keys,
        rows,
        next: 0,
        current: None,
        empty: Vec::new(),
        failed: false,
        yielder: Yielder::ambient(),
    };
    let mut chunked = Chunked::new(fold, Arc::clone(inbox.control()), scope);
    let result = backend.compact_streamed(upto, &mut chunked);
    let (verdict, fold) = chunked.finish(result);
    match verdict {
        Verdict::Committed(()) => Verdict::Committed(mem::take(&mut { fold }.empty)),
        Verdict::Truncated => Verdict::Truncated,
        Verdict::Failed(error) => Verdict::Failed(error),
        Verdict::Cancelled => Verdict::Cancelled,
        Verdict::Panicked(payload) => Verdict::Panicked(payload),
    }
}

/// The flight's snapshot, folded key by key as the backend pulls it, as
/// `SnapshotFold` folds the live store. Stops after the first `Err`.
struct FlightFold<'a> {
    keys: &'a [(String, Option<Arc<LoggedKey>>)],
    rows: Vec<Option<Box<dyn RowBatch>>>,
    next: usize,
    current: Option<(usize, Folded<'a>, bool)>,
    /// The jobs whose key is logged and folded to nothing.
    empty: Vec<usize>,
    failed: bool,
    yielder: Yielder<'static>,
}

impl<'a> FlightFold<'a> {
    fn fold_key(&mut self, job: usize) -> Result<Folded<'a>> {
        let keys = self.keys;
        let (key, logged) = &keys[job];
        let logged = logged.as_deref();
        match self.rows[job].take() {
            Some(rows) => rows.fold(key, logged, &self.yielder),
            None => Ok(fold_logged(key, logged)),
        }
    }
}

impl Iterator for FlightFold<'_> {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Result<Record>> {
        self.yielder.tick();
        while !self.failed {
            if let Some((job, records, pulled)) = &mut self.current {
                match records.next() {
                    Some(Ok(record)) => {
                        *pulled = true;
                        return Some(Ok(record));
                    }
                    Some(Err(e)) => {
                        self.failed = true;
                        return Some(Err(e));
                    }
                    None => {
                        if !*pulled && self.keys[*job].1.is_some() {
                            self.empty.push(*job);
                        }
                        self.current = None;
                    }
                }
            }
            let job = self.next;
            if job == self.keys.len() {
                return None;
            }
            self.next += 1;
            match self.fold_key(job) {
                Ok(records) => self.current = Some((job, records, false)),
                Err(e) => {
                    self.failed = true;
                    return Some(Err(e));
                }
            }
        }
        None
    }
}
