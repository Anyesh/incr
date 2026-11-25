use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize, Serializer};

use super::flight::{StorePark, STORE_CHECKPOINT_PARK};
use super::logged::LoggedKey;
use super::{
    Durability, Input, Options, PersistedCollection, PersistedKeyed, Store, StoreCheckpoint,
};
use crate::value_context::wait_probe::{self, WaitProbe};
use crate::{
    Backend, ConditionalAppend, Error, FileBackend, Kind, Local, MemoryBackend, Position,
    ReadOutcome, Record, Result, SnapshotId,
};

const WAIT: Duration = Duration::from_secs(20);
const GATE_TIMEOUT: Duration = Duration::from_secs(5);
const STEP: Duration = Duration::ZERO;

/// Fails to encode on a checkpoint worker when at or above this.
const TOUCHY: u32 = 1_000_000;

/// A value that encodes everywhere except on a checkpoint worker, where
/// one at or above [`TOUCHY`] fails.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
struct Touchy(u32);

impl Serialize for Touchy {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        if self.0 >= TOUCHY && on_worker() {
            return Err(serde::ser::Error::custom("refused on the worker"));
        }
        s.serialize_u32(self.0)
    }
}

type Rows = PersistedKeyed<u32, String, Local>;
type Snapshot = Option<(Position, Vec<Record>)>;
type Pull = dyn Fn(&MemoryBackend, Position, &mut dyn Iterator<Item = Result<Record>>) -> Result<()>
    + Send
    + Sync;

#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }

    /// Waits until opened, or for [`GATE_TIMEOUT`].
    fn wait(&self) {
        self.wait_within(GATE_TIMEOUT);
    }

    /// Whether the gate opened within `timeout`.
    fn wait_within(&self, timeout: Duration) -> bool {
        let open = self.open.lock().unwrap();
        *self
            .changed
            .wait_timeout_while(open, timeout, |open| !*open)
            .unwrap()
            .0
    }
}

fn on_worker() -> bool {
    std::thread::current().name() == Some("incr-checkpoint")
}

/// A gate that a [`Wrap`] opens when the store asks for its snapshot id
/// once armed.
#[derive(Default)]
struct Asked {
    armed: AtomicBool,
    gate: Gate,
}

/// A [`MemoryBackend`] whose compaction may wait at a gate, on a checkpoint
/// worker only, or pull as a test says, which may decline concurrent
/// compaction, whose reads may stop at a position, and which may tell when
/// the store asks for its snapshot id.
struct Wrap {
    inner: MemoryBackend,
    concurrent: bool,
    gate: Option<Arc<Gate>>,
    pull: Option<Box<Pull>>,
    read_upto: Option<Arc<AtomicU64>>,
    asked: Option<Arc<Asked>>,
}

impl Wrap {
    fn new(inner: &MemoryBackend) -> Self {
        Wrap {
            inner: inner.clone(),
            concurrent: true,
            gate: None,
            pull: None,
            read_upto: None,
            asked: None,
        }
    }

    fn gated(inner: &MemoryBackend, gate: &Arc<Gate>) -> Self {
        Wrap {
            gate: Some(Arc::clone(gate)),
            ..Wrap::new(inner)
        }
    }

    fn pulling(
        inner: &MemoryBackend,
        pull: impl Fn(&MemoryBackend, Position, &mut dyn Iterator<Item = Result<Record>>) -> Result<()>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Wrap {
            pull: Some(Box::new(pull)),
            ..Wrap::new(inner)
        }
    }
}

impl Backend for Wrap {
    fn append(&self, records: &[Record]) -> Result<Position> {
        self.inner.append(records)
    }

    fn sync(&self, upto: Position) -> Result<()> {
        self.inner.sync(upto)
    }

    fn read_from(&self, after: Position, out: &mut Vec<(Position, Record)>) -> Result<ReadOutcome> {
        let outcome = self.inner.read_from(after, out)?;
        if let Some(upto) = &self.read_upto {
            let upto = Position(upto.load(Ordering::SeqCst));
            out.retain(|(position, _)| *position <= upto);
        }
        Ok(outcome)
    }

    fn load_snapshot(&self) -> Result<Option<(Position, Vec<Record>)>> {
        self.inner.load_snapshot()
    }

    fn snapshot_id(&self) -> Result<Option<SnapshotId>> {
        if let Some(asked) = self.asked.as_ref().filter(|_| !on_worker()) {
            if asked.armed.load(Ordering::SeqCst) {
                asked.gate.open();
            }
        }
        self.inner.snapshot_id()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> Result<()> {
        self.compact_streamed(upto, &mut folded.into_iter().map(Ok))
    }

    fn compact_streamed(
        &self,
        upto: Position,
        records: &mut dyn Iterator<Item = Result<Record>>,
    ) -> Result<()> {
        if let Some(gate) = self.gate.as_ref().filter(|_| on_worker()) {
            gate.wait();
        }
        match &self.pull {
            Some(pull) => pull(&self.inner, upto, records),
            None => self.inner.compact_streamed(upto, records),
        }
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }

    fn concurrent_compaction(&self) -> bool {
        self.concurrent
    }

    fn conditional_append(&self) -> Option<&dyn ConditionalAppend> {
        self.inner.conditional_append()
    }
}

/// A store with an input set, an input at its default, a keyed collection
/// and a plain one, over `backend`, which writes to `log`.
struct Fx<B: Backend> {
    log: MemoryBackend,
    store: Store<Local, B>,
    x: Input<u32>,
    y: Input<u32>,
    rows: Rows,
    plain: PersistedCollection<String, Local>,
    touchy: PersistedCollection<Touchy, Local>,
}

impl Fx<MemoryBackend> {
    fn memory() -> Self {
        let log = MemoryBackend::new();
        Fx::over(&log, log.clone(), Options::default())
    }
}

impl<B: Backend + 'static> Fx<B> {
    fn over(log: &MemoryBackend, backend: B, options: Options) -> Self {
        let store = Store::open(backend, options).unwrap();
        Fx {
            log: log.clone(),
            x: store.input("x", 0).unwrap(),
            y: store.input("y", 5).unwrap(),
            rows: store.keyed_collection("rows").unwrap(),
            plain: store.collection("plain").unwrap(),
            touchy: store.collection("touchy").unwrap(),
            store,
        }
    }

    /// `n` keyed rows, some removed and some left at the unlogged rank by
    /// a scope, plain elements with multiplicities, an input set, and
    /// keys no binding holds: one with elements, one at its default and
    /// one set.
    fn populate(&self, n: u32) {
        self.fill(n);
        let store = &self.store;
        self.log
            .append(&[
                declare("ghost", Kind::Collection, &[]),
                insert("ghost", &enc("g1")),
                insert("ghost", &enc("g2")),
                insert("ghost", &enc("g1")),
                declare("idle", Kind::Input, b"d"),
                declare("held", Kind::Input, b"d"),
                Record::Set {
                    key: "held".to_owned(),
                    value: b"v".to_vec(),
                },
            ])
            .unwrap();
        store.refresh().unwrap();
    }

    /// [`Self::populate`] without the keys no binding holds, all through
    /// the store.
    fn fill(&self, n: u32) {
        let store = &self.store;
        store.set(self.x, 7).unwrap();
        for k in 0..n {
            store.upsert(&self.rows, k, format!("row{k}")).unwrap();
        }
        for k in (0..n).step_by(7) {
            store.remove(&self.rows, &k).unwrap();
        }
        for v in ["b", "a", "c", "a", "dd", "e", "a", "ff", "e"] {
            self.plain.insert(store, v.to_owned()).unwrap();
        }
        self.plain.delete(store, &"c".to_owned()).unwrap();
        self.touchy.insert(store, Touchy(3)).unwrap();
        let rt = store.runtime();
        let scope = store.unlogged_scope().unwrap();
        let overlay = scope.keyed(&self.rows).unwrap();
        for k in [3, 17, 29].into_iter().filter(|&k| k < n) {
            overlay.remove(rt, &k);
            overlay.upsert(rt, k, format!("row{k}"));
        }
        scope.close().unwrap();
    }

    fn snapshot(&self) -> Snapshot {
        self.log.load_snapshot().unwrap()
    }

    fn begin(&self) -> StoreCheckpoint {
        self.store.begin_checkpoint().unwrap()
    }

    fn finish(&self, flight: &StoreCheckpoint) -> Position {
        flight
            .result(&self.store, Some(WAIT))
            .unwrap()
            .expect("the flight finishes")
    }
}

#[derive(Debug, PartialEq)]
struct Contents {
    x: u32,
    y: u32,
    rows: Vec<(u32, String)>,
    plain: Vec<String>,
    touchy: Vec<Touchy>,
    ghost: Vec<String>,
    keys: Vec<String>,
}

/// What a store reopened on `log` holds. Leaves out `idle`, which a
/// snapshot forgets at its default and a log without one lists.
fn reopened(log: &MemoryBackend) -> Contents {
    reopened_fx(&Fx::over(log, log.clone(), Options::default()))
}

/// [`reopened`] for a store just opened on any backend.
fn reopened_fx<B: Backend + 'static>(fx: &Fx<B>) -> Contents {
    let mut keys = fx.store.restored_keys("").unwrap();
    keys.retain(|key| key != "idle");
    let ghost = fx.store.collection::<String>("ghost").unwrap();
    let mut ghost = ghost.handle().snapshot_values();
    ghost.sort();
    let mut contents = contents(fx);
    contents.ghost = ghost;
    contents.keys = keys;
    contents
}

fn contents<B: Backend>(fx: &Fx<B>) -> Contents {
    let rt = fx.store.runtime();
    let mut rows = fx.rows.items();
    rows.sort();
    let mut plain = fx.plain.handle().snapshot_values();
    plain.sort();
    let mut touchy = fx.touchy.handle().snapshot_values();
    touchy.sort();
    Contents {
        x: rt.get(fx.x.handle()),
        y: rt.get(fx.y.handle()),
        rows,
        plain,
        touchy,
        ghost: Vec::new(),
        keys: Vec::new(),
    }
}

fn declare(key: &str, kind: Kind, default: &[u8]) -> Record {
    Record::Declare {
        key: key.to_owned(),
        kind,
        default: default.to_vec(),
    }
}

fn enc(element: &str) -> Vec<u8> {
    postcard::to_stdvec(&element.to_owned()).unwrap()
}

fn insert(key: &str, element: &[u8]) -> Record {
    Record::Insert {
        key: key.to_owned(),
        element: element.to_vec(),
    }
}

/// The snapshot a blocking checkpoint writes after `before`, on a store
/// of its own.
fn twin_snapshot(before: impl Fn(&Fx<MemoryBackend>)) -> Snapshot {
    let fx = Fx::memory();
    before(&fx);
    fx.store.checkpoint().unwrap();
    fx.snapshot()
}

/// What a store that never checkpointed holds after `writes`, reopened.
fn control(writes: impl Fn(&Fx<MemoryBackend>)) -> Contents {
    let fx = Fx::memory();
    writes(&fx);
    reopened(&fx.log)
}

/// Runs `before`, a flight with `during` between its begin and its end,
/// and checks the snapshot against a twin's blocking checkpoint after
/// `before`, and a reopen against a control that ran both without
/// checkpointing.
fn fly_and_compare(
    before: impl Fn(&Fx<MemoryBackend>),
    during: impl Fn(&Fx<MemoryBackend>, Option<&StoreCheckpoint>),
) {
    let fx = Fx::memory();
    before(&fx);
    let flight = fx.begin();
    during(&fx, Some(&flight));
    let at = fx.finish(&flight);
    assert!(!fx.store.capture_installed());
    let snapshot = fx.snapshot();
    assert_eq!(snapshot.as_ref().map(|(at, _)| *at), Some(at));
    assert_eq!(snapshot, twin_snapshot(&before));
    assert_eq!(
        reopened(&fx.log),
        control(|fx| {
            before(fx);
            during(fx, None);
        })
    );
}

fn poll(fx: &Fx<MemoryBackend>, flight: Option<&StoreCheckpoint>) {
    if let Some(flight) = flight {
        flight.poll(&fx.store, STEP).unwrap();
    }
}

#[test]
fn a_flight_writes_the_snapshot_a_blocking_checkpoint_writes() {
    let fx = Fx::memory();
    fx.populate(300);
    let flight = fx.begin();
    let at = fx.finish(&flight);
    assert_eq!(fx.store.stats().unwrap().snapshot_position, at);
    assert_eq!(fx.snapshot(), twin_snapshot(|fx| fx.populate(300)));
    let snapshot = fx.snapshot().unwrap().1;
    assert!(
        !snapshot.iter().any(|record| record.key() == "idle"),
        "a key at its default is forgotten"
    );
    assert!(snapshot.iter().any(|record| record.key() == "ghost"));
    assert!(snapshot.iter().any(|record| record.key() == "held"));
}

#[test]
fn edits_between_begin_and_finish_stay_out_of_the_snapshot() {
    fly_and_compare(
        |fx| fx.populate(10_000),
        |fx, flight| {
            let store = &fx.store;
            poll(fx, flight);
            store.set(fx.x, 8).unwrap();
            store.set(fx.y, 6).unwrap();
            store.upsert(&fx.rows, 1, "changed".to_owned()).unwrap();
            store.remove(&fx.rows, &2).unwrap();
            store.upsert(&fx.rows, 20_000, "new".to_owned()).unwrap();
            poll(fx, flight);
            fx.plain.insert(store, "a".to_owned()).unwrap();
            fx.plain.delete(store, &"b".to_owned()).unwrap();
            fx.plain.insert(store, "zz".to_owned()).unwrap();
            poll(fx, flight);
            store.upsert(&fx.rows, 9_999, "late".to_owned()).unwrap();
        },
    );
}

#[test]
fn checkpoint_and_begin_are_busy_while_a_flight_is_open_and_auto_checkpoint_waits() {
    let log = MemoryBackend::new();
    let options = Options {
        auto_checkpoint_records: Some(2),
        ..Options::default()
    };
    let fx = Fx::over(&log, log.clone(), options);
    fx.populate(50);
    let flight = fx.begin();
    assert!(matches!(fx.store.checkpoint(), Err(Error::Busy)));
    assert!(matches!(fx.store.begin_checkpoint(), Err(Error::Busy)));
    let before = fx.store.stats().unwrap();
    for v in 10..15 {
        fx.store.set(fx.x, v).unwrap();
    }
    let during = fx.store.stats().unwrap();
    assert_eq!(during.snapshot_position, before.snapshot_position);
    assert_eq!(
        during.records_since_checkpoint,
        before.records_since_checkpoint + 5
    );
    let at = fx.finish(&flight);
    assert_eq!(at, before.position);
    let settled = fx.store.stats().unwrap();
    assert_eq!(settled.snapshot_position, at);
    assert_eq!(settled.records_since_checkpoint, 5);
    let last = fx.store.stats().unwrap().position;
    fx.store.set(fx.x, 20).unwrap();
    assert_eq!(fx.store.stats().unwrap().snapshot_position, last);
}

/// Begins a flight on a gated backend, seals it, then runs `end` and lets
/// the worker go; drops the handle once the verdict is in and checks that
/// the first plain write settles it, adopting the flight's snapshot only
/// when `adopted`, that writes run the auto checkpoint when due, and that a
/// checkpoint then succeeds.
fn a_dropped_handle_settles_at_the_next_write(
    adopted: bool,
    end: impl FnOnce(&Fx<Wrap>, &StoreCheckpoint),
) {
    let log = MemoryBackend::new();
    let gate = Arc::new(Gate::default());
    let options = Options {
        auto_checkpoint_records: Some(3),
        ..Options::default()
    };
    let fx = Fx::over(&log, Wrap::gated(&log, &gate), options);
    fx.populate(50);
    let id_before = log.snapshot_id().unwrap();
    let flight = fx.begin();
    assert!(!flight.poll(&fx.store, WAIT).unwrap());
    end(&fx, &flight);
    gate.open();
    assert!(fx.store.wait_flight_verdict(WAIT));
    let id_after = log.snapshot_id().unwrap();
    assert_eq!(id_after != id_before, adopted);
    drop(flight);
    let begun = fx.store.stats().unwrap();
    fx.store.set(fx.x, 100).unwrap();
    assert!(!fx.store.capture_installed());
    if adopted {
        assert_eq!(
            fx.store.stats().unwrap().snapshot_position,
            begun.position,
            "the write adopts the flight's snapshot"
        );
        assert_eq!(
            log.snapshot_id().unwrap(),
            id_after,
            "and runs no checkpoint of its own"
        );
    }
    let mut ran = false;
    for v in 101..110 {
        fx.store.set(fx.x, v).unwrap();
        if fx.store.stats().unwrap().snapshot_position > begun.position {
            ran = true;
            break;
        }
    }
    assert!(ran, "the auto checkpoint runs once due");
    let at = fx.store.checkpoint().unwrap();
    assert_eq!(fx.store.stats().unwrap().snapshot_position, at);
    assert_eq!(reopened(&fx.log), reopened_of(&fx));
}

/// What `fx`'s own store holds, in [`reopened`]'s terms.
fn reopened_of<B: Backend>(fx: &Fx<B>) -> Contents {
    let mut contents = contents(fx);
    let mut keys = fx.store.restored_keys("").unwrap();
    keys.retain(|key| key != "idle");
    keys.sort();
    let mut reopened = reopened(&fx.log);
    reopened.keys.sort();
    contents.keys = keys;
    contents.ghost = reopened.ghost;
    contents
}

#[test]
fn a_handle_dropped_after_the_commit_settles_at_the_next_write() {
    a_dropped_handle_settles_at_the_next_write(true, |_, _| {});
}

#[test]
fn a_handle_dropped_after_a_cancel_settles_at_the_next_write() {
    a_dropped_handle_settles_at_the_next_write(false, |_, flight| flight.cancel());
}

#[test]
fn a_handle_dropped_after_a_worker_error_settles_at_the_next_write() {
    let log = MemoryBackend::new();
    let gate = Arc::new(Gate::default());
    let options = Options {
        auto_checkpoint_records: Some(3),
        ..Options::default()
    };
    let fx = Fx::over(&log, Wrap::gated(&log, &gate), options);
    fx.populate(50);
    fx.touchy.insert(&fx.store, Touchy(TOUCHY + 1)).unwrap();
    let flight = fx.begin();
    assert!(!flight.poll(&fx.store, WAIT).unwrap());
    gate.open();
    assert!(fx.store.wait_flight_verdict(WAIT));
    drop(flight);
    let begun = fx.store.stats().unwrap();
    fx.store.set(fx.x, 100).unwrap();
    assert!(!fx.store.capture_installed());
    assert_eq!(
        fx.store.stats().unwrap().snapshot_position,
        begun.snapshot_position
    );
    fx.touchy.delete(&fx.store, &Touchy(TOUCHY + 1)).unwrap();
    let mut ran = false;
    for v in 101..110 {
        fx.store.set(fx.x, v).unwrap();
        if fx.store.stats().unwrap().snapshot_position > begun.position {
            ran = true;
            break;
        }
    }
    assert!(ran, "the auto checkpoint runs once due");
    let at = fx.store.checkpoint().unwrap();
    assert_eq!(fx.store.stats().unwrap().snapshot_position, at);
}

#[test]
fn a_flight_on_a_backend_without_concurrent_compaction_is_refused() {
    let log = MemoryBackend::new();
    let fx = Fx::over(
        &log,
        Wrap {
            concurrent: false,
            ..Wrap::new(&log)
        },
        Options::default(),
    );
    fx.populate(10);
    assert!(matches!(
        fx.store.begin_checkpoint(),
        Err(Error::ConcurrentCompactionUnsupported)
    ));
    fx.store.checkpoint().unwrap();
}

/// A flight over a backend that pulls with `pull`, and its verdict.
fn judged(
    n: u32,
    pull: impl Fn(&MemoryBackend, Position, &mut dyn Iterator<Item = Result<Record>>) -> Result<()>
        + Send
        + Sync
        + 'static,
) -> (Fx<Wrap>, Result<Option<Position>>) {
    let log = MemoryBackend::new();
    let fx = Fx::over(&log, Wrap::pulling(&log, pull), Options::default());
    fx.populate(n);
    let flight = fx.begin();
    let result = flight.result(&fx.store, Some(WAIT));
    (fx, result)
}

fn assert_poisoned(fx: &Fx<Wrap>, result: Result<Option<Position>>) {
    assert!(matches!(result, Err(Error::Corrupt(_))), "{result:?}");
    assert!(matches!(fx.store.stats(), Err(Error::Poisoned)));
}

#[test]
fn judge_a_backend_returning_after_one_record_poisons() {
    let (fx, result) = judged(600, |_, _, records| {
        records.next();
        Ok(())
    });
    assert_poisoned(&fx, result);
}

#[test]
fn judge_a_backend_returning_after_exactly_256_records_poisons() {
    let (fx, result) = judged(600, |_, _, records| {
        for _ in 0..256 {
            records.next();
        }
        Ok(())
    });
    assert_poisoned(&fx, result);
}

#[test]
fn judge_a_backend_taking_every_record_without_the_end_is_adopted() {
    let log = MemoryBackend::new();
    let store = Store::<Local, _>::open(
        Wrap::pulling(&log, |inner, upto, records| {
            let folded = records.take(512).collect::<Result<Vec<_>>>()?;
            assert_eq!(folded.len(), 512);
            inner.compact(upto, folded)
        }),
        Options::default(),
    )
    .unwrap();
    let rows: Rows = store.keyed_collection("rows").unwrap();
    for k in 0..511 {
        store.upsert(&rows, k, format!("row{k}")).unwrap();
    }
    let flight = store.begin_checkpoint().unwrap();
    let at = flight.result(&store, Some(WAIT)).unwrap().unwrap();
    let stats = store.stats().unwrap();
    assert_eq!(stats.snapshot_position, at);
    assert_eq!(log.load_snapshot().unwrap().unwrap().1.len(), 512);
}

#[test]
fn judge_a_backend_swallowing_a_failed_pull_poisons() {
    let log = MemoryBackend::new();
    let fx = Fx::over(
        &log,
        Wrap::pulling(&log, |_, _, records| {
            for record in records {
                if record.is_err() {
                    break;
                }
            }
            Ok(())
        }),
        Options::default(),
    );
    fx.populate(20);
    fx.touchy.insert(&fx.store, Touchy(TOUCHY + 2)).unwrap();
    let flight = fx.begin();
    let result = flight.result(&fx.store, Some(WAIT));
    assert_poisoned(&fx, result);
}

#[test]
fn judge_a_cancel_at_a_pull_reaches_the_backend_as_an_error() {
    let log = MemoryBackend::new();
    let pulled = Arc::new(Gate::default());
    let go = Arc::new(Gate::default());
    let saw_cancel = Arc::new(AtomicBool::new(false));
    let backend = {
        let (pulled, go, saw_cancel) = (
            Arc::clone(&pulled),
            Arc::clone(&go),
            Arc::clone(&saw_cancel),
        );
        Wrap::pulling(&log, move |inner, upto, records| {
            let first = records.next();
            assert!(matches!(first, Some(Ok(_))));
            pulled.open();
            go.wait();
            let next = records.next();
            saw_cancel.store(
                matches!(next, Some(Err(Error::Cancelled))),
                Ordering::SeqCst,
            );
            next.transpose()?;
            inner.compact_streamed(upto, records)
        })
    };
    let fx = Fx::over(&log, backend, Options::default());
    fx.populate(20);
    let flight = fx.begin();
    assert!(!flight.poll(&fx.store, WAIT).unwrap());
    pulled.wait();
    flight.cancel();
    go.open();
    assert!(matches!(
        flight.result(&fx.store, Some(WAIT)),
        Err(Error::Cancelled)
    ));
    assert!(saw_cancel.load(Ordering::SeqCst));
    assert!(fx.snapshot().is_none());
    assert!(!fx.store.capture_installed());
    fx.store.checkpoint().unwrap();
}

#[test]
fn capture_a_keyed_rehash_mid_capture() {
    fly_and_compare(
        |fx| fx.populate(5_000),
        |fx, flight| {
            poll(fx, flight);
            for k in 5_000..15_000 {
                fx.store.upsert(&fx.rows, k, format!("row{k}")).unwrap();
                if k % 1_000 == 0 {
                    let old = (k - 5_000) / 2;
                    fx.store.upsert(&fx.rows, old, format!("moved{k}")).unwrap();
                }
            }
            poll(fx, flight);
            fx.store.remove(&fx.rows, &4).unwrap();
            fx.store.upsert(&fx.rows, 4_999, "last".to_owned()).unwrap();
        },
    );
}

#[test]
fn capture_a_plain_rehash_mid_capture() {
    fly_and_compare(
        |fx| {
            fx.populate(10);
            for v in 0..5_000 {
                fx.plain.insert(&fx.store, format!("p{v}")).unwrap();
            }
        },
        |fx, flight| {
            poll(fx, flight);
            for v in 5_000..15_000 {
                fx.plain.insert(&fx.store, format!("p{v}")).unwrap();
            }
            poll(fx, flight);
            fx.plain.delete(&fx.store, &"p1".to_owned()).unwrap();
            fx.plain.insert(&fx.store, "p2".to_owned()).unwrap();
        },
    );
}

/// The distinct rows or elements that fill a table of 8,192 slots to its
/// load limit: the next one starts a migration into 16,384 slots, which
/// 32 single-key writes finish.
const TABLE_FULL: u32 = 6_144;

fn index_bytes(fx: &Fx<MemoryBackend>) -> usize {
    let index = fx.rows.index_weak().upgrade().expect("the index is alive");
    index.settled().1
}

/// Leaves the key table and the pairs' table six writes into a migration.
fn half_migrated_rows(fx: &Fx<MemoryBackend>) {
    for k in 0..TABLE_FULL {
        fx.store.upsert(&fx.rows, k, format!("row{k}")).unwrap();
    }
    let one_table = index_bytes(fx);
    for k in TABLE_FULL..TABLE_FULL + 6 {
        fx.store.upsert(&fx.rows, k, format!("row{k}")).unwrap();
    }
    assert_eq!(
        index_bytes(fx),
        3 * one_table,
        "the key table still holds its old storage"
    );
}

#[test]
fn capture_a_keyed_table_begun_mid_migration() {
    fly_and_compare(half_migrated_rows, |fx, flight| {
        poll(fx, flight);
        for k in (0..5_000).step_by(500) {
            fx.store.upsert(&fx.rows, k, format!("moved{k}")).unwrap();
            fx.store.remove(&fx.rows, &(k + 1)).unwrap();
        }
        poll(fx, flight);
        for k in 7_000..7_040 {
            fx.store.upsert(&fx.rows, k, format!("row{k}")).unwrap();
        }
        poll(fx, flight);
        fx.store.remove(&fx.rows, &4).unwrap();
        fx.store.upsert(&fx.rows, 6_143, "last".to_owned()).unwrap();
    });
}

#[test]
fn capture_a_plain_table_begun_mid_migration() {
    fly_and_compare(
        |fx| {
            for v in 0..TABLE_FULL + 6 {
                fx.plain.insert(&fx.store, format!("p{v}")).unwrap();
            }
        },
        |fx, flight| {
            poll(fx, flight);
            for v in (0..5_000).step_by(500) {
                fx.plain.delete(&fx.store, &format!("p{v}")).unwrap();
                fx.plain.insert(&fx.store, format!("p{}", v + 1)).unwrap();
            }
            poll(fx, flight);
            for v in 7_000..7_040 {
                fx.plain.insert(&fx.store, format!("p{v}")).unwrap();
            }
            poll(fx, flight);
            fx.plain.delete(&fx.store, &"p4".to_owned()).unwrap();
            fx.plain.insert(&fx.store, "p6143".to_owned()).unwrap();
        },
    );
}

#[test]
fn capture_an_unlogged_scope_across_polls() {
    fly_and_compare(
        |fx| fx.populate(10_000),
        |fx, flight| {
            poll(fx, flight);
            let rt = fx.store.runtime();
            let scope = fx.store.unlogged_scope().unwrap();
            let overlay = scope.keyed(&fx.rows).unwrap();
            for k in [1, 8_000, 9_001] {
                overlay.remove(rt, &k);
            }
            poll(fx, flight);
            for k in [1, 8_000, 9_001] {
                overlay.upsert(rt, k, format!("row{k}"));
            }
            poll(fx, flight);
            scope.close().unwrap();
            poll(fx, flight);
        },
    );
}

#[test]
fn capture_a_staged_batch_across_polls() {
    fly_and_compare(
        |fx| fx.populate(10_000),
        |fx, flight| {
            poll(fx, flight);
            let mut batch = fx.store.staged_batch().unwrap();
            batch.set(fx.x, 99).unwrap();
            batch.insert(&fx.plain, "staged".to_owned()).unwrap();
            poll(fx, flight);
            batch.delete(&fx.plain, "a".to_owned()).unwrap();
            batch.commit().unwrap();
            poll(fx, flight);
            fx.store.upsert(&fx.rows, 3, "after".to_owned()).unwrap();
        },
    );
}

#[test]
fn capture_an_unbound_logged_key_written_mid_flight() {
    fly_and_compare(
        |fx| fx.populate(10_000),
        |fx, flight| {
            poll(fx, flight);
            fx.log
                .append(&[
                    insert("ghost", &enc("g3")),
                    Record::Remove {
                        key: "ghost".to_owned(),
                        element: enc("g1"),
                    },
                    Record::Set {
                        key: "idle".to_owned(),
                        value: b"w".to_vec(),
                    },
                    Record::Delete {
                        key: "held".to_owned(),
                    },
                ])
                .unwrap();
            fx.store.refresh().unwrap();
            poll(fx, flight);
        },
    );
}

#[test]
fn capture_undeclare_mid_capture() {
    fly_and_compare(
        |fx| fx.populate(10_000),
        |fx, flight| {
            poll(fx, flight);
            fx.store.undeclare(&fx.rows).unwrap();
            fx.store.undeclare(&fx.plain).unwrap();
            poll(fx, flight);
            let rows: Rows = fx.store.keyed_collection("rows").unwrap();
            fx.store.upsert(&rows, 2, "rebound".to_owned()).unwrap();
        },
    );
}

#[test]
fn a_store_poisoned_mid_flight_holds_no_capture_state() {
    let fx = Fx::memory();
    fx.populate(10_000);
    let flight = fx.begin();
    assert!(!flight.poll(&fx.store, STEP).unwrap());
    assert!(fx.store.capture_installed());
    let scope = fx.store.unlogged_scope().unwrap();
    drop(scope);
    assert!(matches!(fx.store.stats(), Err(Error::Poisoned)));
    assert!(!fx.store.capture_installed());
    assert!(flight.poll(&fx.store, STEP).is_err());
}

#[test]
fn a_rebuild_mid_flight_cancels_it() {
    let fx = Fx::memory();
    fx.populate(10_000);
    let other = Fx::over(&fx.log, fx.log.clone(), Options::default());
    let flight = fx.begin();
    assert!(!flight.poll(&fx.store, STEP).unwrap());
    other.store.set(other.x, 42).unwrap();
    other.store.checkpoint().unwrap();
    fx.store.refresh().unwrap();
    assert!(matches!(
        flight.result(&fx.store, Some(WAIT)),
        Err(Error::Cancelled)
    ));
    assert!(!fx.store.capture_installed());
    assert_eq!(fx.store.runtime().get(fx.x.handle()), 42);
    fx.store.checkpoint().unwrap();
    assert_eq!(reopened(&fx.log), reopened_of(&fx));
}

#[test]
fn a_flight_cancelled_by_a_rebuild_is_not_adopted_when_its_worker_still_commits() {
    let log = MemoryBackend::new();
    let pulled = Arc::new(Gate::default());
    let go = Arc::new(Gate::default());
    let committed = Arc::new(Gate::default());
    let backend = {
        let (pulled, go, committed) =
            (Arc::clone(&pulled), Arc::clone(&go), Arc::clone(&committed));
        Wrap::pulling(&log, move |inner, upto, records| {
            let folded = records.collect::<Result<Vec<_>>>()?;
            pulled.open();
            go.wait();
            let done = inner.compact(upto, folded);
            committed.open();
            done
        })
    };
    let fx = Fx::over(&log, backend, Options::default());
    fx.populate(100);
    let other = Second::lagging(&log);
    fx.store.set(fx.x, 8).unwrap();
    let flight = fx.begin();
    assert!(!flight.poll(&fx.store, WAIT).unwrap());
    pulled.wait();
    other.store.checkpoint().unwrap();
    fx.store.refresh().unwrap();
    let rebuilt = log.snapshot_id().unwrap();
    go.open();
    committed.wait();
    assert_ne!(log.snapshot_id().unwrap(), rebuilt, "the worker committed");
    assert!(matches!(
        flight.result(&fx.store, Some(WAIT)),
        Err(Error::Cancelled)
    ));
    assert!(!fx.store.capture_installed());
    fx.store.set(fx.x, 9).unwrap();
    fx.store.checkpoint().unwrap();
    assert_eq!(reopened(&fx.log), reopened_of(&fx));
}

/// A second store on a log, which reads no further than the position it
/// opened at until [`Second::lift`].
struct Second {
    store: Store<Local, Wrap>,
    read_upto: Arc<AtomicU64>,
}

impl Second {
    fn lagging(log: &MemoryBackend) -> Self {
        let read_upto = Arc::new(AtomicU64::new(u64::MAX));
        let backend = Wrap {
            read_upto: Some(Arc::clone(&read_upto)),
            ..Wrap::new(log)
        };
        let store = Store::open(backend, Options::default()).unwrap();
        read_upto.store(store.stats().unwrap().position.0, Ordering::SeqCst);
        Second { store, read_upto }
    }

    fn lift(&self) {
        self.read_upto.store(u64::MAX, Ordering::SeqCst);
    }
}

/// The keys whose fold is empty when a [`Forgetting`] begins its flight:
/// `empty`, a bound collection without elements; `idle`, unbound at its
/// default; `spare`, an unbound collection key; `y`, bound at its default.
const FORGOTTEN: [&str; 4] = ["empty", "idle", "spare", "y"];

/// The default `populate` logs for `idle`, decoded as a `u32`.
const IDLE: u32 = 100;

/// A populated store whose worker waits at `gate`, and a second store on
/// its log that reads one record behind it.
struct Forgetting {
    fx: Fx<Wrap>,
    gate: Arc<Gate>,
    second: Second,
}

impl Forgetting {
    /// Its flight fails on a value the worker cannot encode when `failed`.
    fn new(failed: bool) -> Self {
        let log = MemoryBackend::new();
        let gate = Arc::new(Gate::default());
        let fx = Fx::over(&log, Wrap::gated(&log, &gate), Options::default());
        fx.populate(20);
        fx.store.collection::<String>("empty").unwrap();
        log.append(&[declare("spare", Kind::Collection, &[])])
            .unwrap();
        if failed {
            fx.touchy.insert(&fx.store, Touchy(TOUCHY + 1)).unwrap();
        }
        fx.store.refresh().unwrap();
        let second = Second::lagging(&log);
        fx.store.set(fx.x, 8).unwrap();
        Forgetting { fx, gate, second }
    }

    fn foreign(&self, records: &[Record]) {
        self.fx.log.append(records).unwrap();
        self.fx.store.refresh().unwrap();
    }
}

/// Runs `during` between the begin and the adoption of a flight that
/// commits or, when `failed`, fails, and on a twin after a blocking
/// checkpoint at the flight's position, or none when `failed`. Checks
/// that after the first plain write, and again after a checkpoint, both
/// stores and their reopens hold the forgotten keys alike, and that the
/// two checkpoints write the same snapshot. Returns the flight's result.
fn forget_and_compare(failed: bool, during: impl Fn(&Forgetting)) -> Result<Option<Position>> {
    compare_forgetting(failed, true, during)
}

/// [`forget_and_compare`], comparing the stores' positions only when
/// `positions`. A failed flight's store may log a `Declare` twice, so its
/// positions run ahead of the twin's from the first one on.
fn compare_forgetting(
    failed: bool,
    positions: bool,
    during: impl Fn(&Forgetting),
) -> Result<Option<Position>> {
    let ours = Forgetting::new(failed);
    let twin = Forgetting::new(failed);
    let flight = ours.fx.begin();
    assert!(!flight.poll(&ours.fx.store, WAIT).unwrap());
    if !failed {
        twin.fx.store.checkpoint().unwrap();
    }
    during(&ours);
    during(&twin);
    ours.gate.open();
    let result = flight.result(&ours.fx.store, Some(WAIT));
    assert!(!ours.fx.store.capture_installed());
    for side in [&ours, &twin] {
        side.fx.store.set(side.fx.x, 50).unwrap();
    }
    assert_alike(&ours.fx, &twin.fx, positions, positions && !failed);
    for side in [&ours, &twin] {
        side.fx.store.checkpoint().unwrap();
    }
    assert_alike(&ours.fx, &twin.fx, positions && !failed, positions);
    assert_eq!(
        ours.fx.snapshot().map(|(_, records)| records),
        twin.fx.snapshot().map(|(_, records)| records)
    );
    result
}

#[derive(Debug, PartialEq)]
struct Forgotten {
    logged: Vec<Option<LoggedKey>>,
    idle: Option<u32>,
    empty: Option<Vec<String>>,
    spare: Option<Vec<String>>,
}

/// What `store` holds for the [`FORGOTTEN`] keys: their logged entries,
/// and the values of those it binds.
fn forgotten<B: Backend + 'static>(store: &Store<Local, B>) -> Forgotten {
    let (logged, bound) = {
        let state = store.state.lock().unwrap();
        let logged = FORGOTTEN
            .iter()
            .map(|key| state.logged.get(key).cloned())
            .collect();
        let bound = |key: &str| state.bound.contains_key(key);
        (logged, [bound("idle"), bound("empty"), bound("spare")])
    };
    let elements = |key: &str| {
        let mut elements = store
            .collection::<String>(key)
            .unwrap()
            .handle()
            .snapshot_values();
        elements.sort();
        elements
    };
    Forgotten {
        logged,
        idle: bound[0].then(|| {
            let idle = store.input::<u32>("idle", IDLE).unwrap();
            store.runtime().get(idle.handle())
        }),
        empty: bound[1].then(|| elements("empty")),
        spare: bound[2].then(|| elements("spare")),
    }
}

fn assert_alike(ours: &Fx<Wrap>, twin: &Fx<Wrap>, positions: bool, counts: bool) {
    assert_eq!(forgotten(&ours.store), forgotten(&twin.store), "live");
    assert_eq!(contents(ours), contents(twin));
    let reopen = |log: &MemoryBackend| Store::<Local, _>::open(log.clone(), Options::default());
    assert_eq!(
        forgotten(&reopen(&ours.log).unwrap()),
        forgotten(&reopen(&twin.log).unwrap()),
        "reopened"
    );
    let (ours, twin) = (ours.store.stats().unwrap(), twin.store.stats().unwrap());
    if positions {
        assert_eq!(ours.snapshot_position, twin.snapshot_position);
    }
    if counts {
        assert_eq!(ours.records_since_checkpoint, twin.records_since_checkpoint);
    }
}

fn declare_idle(side: &Forgetting) {
    side.fx.store.input::<u32>("idle", IDLE).unwrap();
}

#[test]
fn forgotten_declare_after_a_committed_flight() {
    forget_and_compare(false, declare_idle).unwrap();
}

#[test]
fn forgotten_declare_after_a_failed_flight() {
    forget_and_compare(true, declare_idle).unwrap_err();
}

fn set_y(side: &Forgetting) {
    side.fx.store.set(side.fx.y, 6).unwrap();
}

#[test]
fn forgotten_set_after_a_committed_flight() {
    forget_and_compare(false, set_y).unwrap();
}

#[test]
fn forgotten_set_after_a_failed_flight() {
    forget_and_compare(true, set_y).unwrap_err();
}

fn delete_idle_and_y(side: &Forgetting) {
    side.fx.store.delete("idle").unwrap();
    side.fx.store.delete(side.fx.y).unwrap();
}

#[test]
fn forgotten_delete_after_a_committed_flight() {
    forget_and_compare(false, delete_idle_and_y).unwrap();
}

#[test]
fn forgotten_delete_after_a_failed_flight() {
    forget_and_compare(true, delete_idle_and_y).unwrap_err();
}

fn foreign_set_and_insert(side: &Forgetting) {
    side.foreign(&[
        Record::Set {
            key: "idle".to_owned(),
            value: postcard::to_stdvec(&7u32).unwrap(),
        },
        insert("spare", &enc("s1")),
    ]);
}

#[test]
fn forgotten_foreign_record_after_a_committed_flight() {
    forget_and_compare(false, foreign_set_and_insert).unwrap();
}

#[test]
fn forgotten_foreign_record_after_a_failed_flight() {
    forget_and_compare(true, foreign_set_and_insert).unwrap_err();
}

fn write_and_undeclare_empty(side: &Forgetting) {
    let store = &side.fx.store;
    let empty = store.collection::<String>("empty").unwrap();
    empty.insert(store, "e1".to_owned()).unwrap();
    store.undeclare(&empty).unwrap();
}

#[test]
fn forgotten_empty_collection_written_and_undeclared_after_a_committed_flight() {
    forget_and_compare(false, write_and_undeclare_empty).unwrap();
}

#[test]
fn forgotten_empty_collection_written_and_undeclared_after_a_failed_flight() {
    forget_and_compare(true, write_and_undeclare_empty).unwrap_err();
}

fn bind_write_and_undeclare_idle(side: &Forgetting) {
    let store = &side.fx.store;
    let idle = store.input::<u32>("idle", IDLE).unwrap();
    store.set(idle, 9).unwrap();
    store.undeclare(idle).unwrap();
}

#[test]
fn forgotten_unbound_key_bound_written_and_undeclared_after_a_committed_flight() {
    forget_and_compare(false, bind_write_and_undeclare_idle).unwrap();
}

#[test]
fn forgotten_unbound_key_bound_written_and_undeclared_after_a_failed_flight() {
    forget_and_compare(true, bind_write_and_undeclare_idle).unwrap_err();
}

fn foreign_insert_then_bind_spare(side: &Forgetting) {
    side.foreign(&[insert("spare", &enc("s1"))]);
    side.fx.store.collection::<String>("spare").unwrap();
}

/// The snapshot's `Insert`s of `s1` into `spare`.
fn spare_inserts(fx: &Fx<Wrap>) -> usize {
    let (_, records) = fx.snapshot().unwrap();
    records
        .iter()
        .filter(|record| **record == insert("spare", &enc("s1")))
        .count()
}

#[test]
fn forgotten_foreign_insert_then_bound_after_a_committed_flight() {
    forget_and_compare(false, foreign_insert_then_bind_spare).unwrap();
    let ours = Forgetting::new(false);
    let flight = ours.fx.begin();
    assert!(!flight.poll(&ours.fx.store, WAIT).unwrap());
    foreign_insert_then_bind_spare(&ours);
    ours.gate.open();
    ours.fx.finish(&flight);
    ours.fx.store.checkpoint().unwrap();
    assert_eq!(spare_inserts(&ours.fx), 1);
}

#[test]
fn forgotten_foreign_insert_then_bound_after_a_failed_flight() {
    forget_and_compare(true, foreign_insert_then_bind_spare).unwrap_err();
}

fn foreign_declare_of_y_with_another_default(side: &Forgetting) {
    side.foreign(&[declare(
        "y",
        Kind::Input,
        &postcard::to_stdvec(&9u32).unwrap(),
    )]);
}

#[test]
fn forgotten_foreign_declare_with_another_default_after_a_committed_flight() {
    forget_and_compare(false, foreign_declare_of_y_with_another_default).unwrap();
    let ours = Forgetting::new(false);
    let flight = ours.fx.begin();
    assert!(!flight.poll(&ours.fx.store, WAIT).unwrap());
    foreign_declare_of_y_with_another_default(&ours);
    ours.gate.open();
    ours.fx.finish(&flight);
    ours.fx.store.set(ours.fx.x, 1).unwrap();
    assert_eq!(ours.fx.store.runtime().get(ours.fx.y.handle()), 9);
    assert_eq!(reopened(&ours.fx.log).y, 9);
}

#[test]
fn forgotten_foreign_declare_with_another_default_after_a_failed_flight() {
    forget_and_compare(true, foreign_declare_of_y_with_another_default).unwrap_err();
}

/// The second store checkpoints after `before`, then declares `idle` with
/// another default, and the first store catches up.
fn second_checkpoints_and_declares(side: &Forgetting, before: impl Fn(&Forgetting)) {
    before(side);
    side.second.store.checkpoint().unwrap();
    side.second.lift();
    side.second.store.input::<u32>("idle", 200).unwrap();
    side.fx.store.refresh().unwrap();
}

#[test]
fn second_store_checkpoint_below_upto_then_declares_a_forgotten_key() {
    let result = compare_forgetting(false, false, |side| {
        second_checkpoints_and_declares(side, |_| {});
    });
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
}

#[test]
fn second_store_checkpoint_at_upto_then_declares_a_forgotten_key() {
    let result = compare_forgetting(false, false, |side| {
        second_checkpoints_and_declares(side, |side| side.second.lift());
    });
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
}

#[test]
fn second_store_checkpoint_above_upto_then_declares_a_forgotten_key() {
    let result = compare_forgetting(false, false, |side| {
        second_checkpoints_and_declares(side, |side| {
            side.fx.store.set(side.fx.x, 9).unwrap();
            side.second.lift();
        });
    });
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
}

fn counting_waits() -> Options {
    Options {
        value_scope: Some(wait_probe::counting_waits()),
        ..Options::default()
    }
}

/// A flight whose backend parks after its commit until the call `meet`
/// makes asks for the backend's snapshot id, then until that call waits
/// inside the value scope's `wait`. `meet` must wait for the verdict there
/// and settle the flight as adopted.
fn a_reconcile_meeting_own_waits_and_adopts(meet: impl FnOnce(&Fx<Wrap>)) {
    let log = MemoryBackend::new();
    let committed = Arc::new(Gate::default());
    let asked = Arc::new(Asked::default());
    let probe = Arc::new(WaitProbe::default());
    let backend = {
        let (committed, asked) = (Arc::clone(&committed), Arc::clone(&asked));
        let probe = Arc::clone(&probe);
        Wrap {
            asked: Some(Arc::clone(&asked)),
            ..Wrap::pulling(&log, move |inner, upto, records| {
                let done = inner.compact_streamed(upto, records);
                committed.open();
                asked.gate.wait();
                probe.record(GATE_TIMEOUT);
                done
            })
        }
    };
    let fx = Fx::over(&log, backend, counting_waits());
    fx.populate(50);
    let begun = fx.store.stats().unwrap().position;
    let flight = fx.begin();
    assert!(!flight.poll(&fx.store, WAIT).unwrap());
    committed.wait();
    asked.armed.store(true, Ordering::SeqCst);
    wait_probe::watch(Some(Arc::clone(&probe)));
    meet(&fx);
    let waits = wait_probe::waits();
    wait_probe::watch(None);
    assert_eq!(waits, 1, "the call entered the scope's wait once");
    assert_eq!(
        probe.seen(),
        Some(true),
        "the worker finished while the call waited inside the scope's wait"
    );
    assert_eq!(
        fx.store.stats().unwrap().snapshot_position,
        begun,
        "the call waited for the verdict and settled it"
    );
    assert_eq!(flight.result(&fx.store, Some(WAIT)).unwrap(), Some(begun));
    fx.store.set(fx.x, 3).unwrap();
    fx.store.checkpoint().unwrap();
    assert_eq!(reopened(&fx.log), reopened_of(&fx));
}

#[test]
fn reconcile_wait_a_catch_up_meeting_own_adopts() {
    a_reconcile_meeting_own_waits_and_adopts(|fx| {
        fx.store.refresh().unwrap();
    });
}

#[test]
fn reconcile_wait_a_declare_meeting_own_adopts() {
    a_reconcile_meeting_own_waits_and_adopts(|fx| {
        fx.store.input::<u32>("idle", IDLE).unwrap();
    });
}

#[test]
fn result_waits_for_the_worker_inside_the_scope_wait() {
    let log = MemoryBackend::new();
    let probe = Arc::new(WaitProbe::default());
    let backend = {
        let probe = Arc::clone(&probe);
        Wrap::pulling(&log, move |inner, upto, records| {
            probe.record(GATE_TIMEOUT);
            inner.compact_streamed(upto, records)
        })
    };
    let fx = Fx::over(&log, backend, counting_waits());
    fx.populate(50);
    let flight = fx.begin();
    wait_probe::watch(Some(Arc::clone(&probe)));
    let at = flight.result(&fx.store, Some(WAIT));
    let waits = wait_probe::waits();
    wait_probe::watch(None);
    assert!(waits > 0, "result entered the scope's wait");
    assert_eq!(
        probe.seen(),
        Some(true),
        "the worker finished while result waited inside the scope's wait"
    );
    assert!(matches!(at, Ok(Some(_))), "{at:?}");
}

const SEEDS: [u64; 4] = [1, 7, 42, 1009];
const SCRIPT_ROWS: u32 = 2_000;
const EDITS: u32 = 48;

/// Edits drawn from a seeded splitmix64: inputs set, the `y` input moved
/// around its default, keyed upserts and removes, plain inserts and
/// deletes, and while `declares` holds, inputs declared under new keys
/// and set.
struct Script {
    state: u64,
    step: u32,
    declares: bool,
    declared: Vec<(String, u32)>,
}

impl Script {
    fn new(seed: u64) -> Self {
        Script {
            state: seed,
            step: 0,
            declares: true,
            declared: Vec::new(),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn edit<B: Backend + 'static>(&mut self, fx: &Fx<B>) {
        let r = self.next();
        let step = self.step;
        self.step += 1;
        let store = &fx.store;
        let small = u32::try_from((r >> 16) % 1_000).unwrap();
        let key = u32::try_from((r >> 16) % u64::from(SCRIPT_ROWS + 50)).unwrap();
        let element =
            ["a", "b", "e", "q", "zz"][usize::try_from((r >> 40) % 5).unwrap()].to_owned();
        match r % 9 {
            0 => store.set(fx.x, small),
            1 => store.set(fx.y, 4 + small % 3),
            2 | 3 => store.upsert(&fx.rows, key, format!("s{step}")),
            4 => store.remove(&fx.rows, &key),
            5 | 6 => fx.plain.insert(store, element),
            7 => fx.plain.delete(store, &element),
            _ if !self.declares => store.set(fx.x, small),
            _ => {
                let name = format!("new{step}");
                let input = store.input(&name, small).unwrap();
                self.declared.push((name, small));
                store.set(input, small + 1)
            }
        }
        .unwrap();
    }

    fn edits<B: Backend + 'static>(&mut self, fx: &Fx<B>, n: u32) {
        for _ in 0..n {
            self.edit(fx);
        }
    }

    /// `seed`'s script applied to `fx` as `runs` say: so many edits with
    /// declares or without.
    fn replayed<B: Backend + 'static>(fx: &Fx<B>, seed: u64, runs: &[(u32, bool)]) -> Script {
        let mut script = Script::new(seed);
        for &(n, declares) in runs {
            script.declares = declares;
            script.edits(fx, n);
        }
        script
    }
}

/// Whether edits made while a flight's worker waits at `at` may declare:
/// not once the backend committed, where a declare waits for the verdict.
fn declares_at(at: StorePark) -> bool {
    at != StorePark::Committed
}

/// What a store just opened holds, and the values of the inputs `script`
/// declared.
fn scripted<B: Backend + 'static>(fx: &Fx<B>, script: &Script) -> (Contents, Vec<u32>) {
    let rt = fx.store.runtime();
    let declared = script
        .declared
        .iter()
        .map(|(name, default)| rt.get(fx.store.input(name, *default).unwrap().handle()))
        .collect();
    (reopened_fx(fx), declared)
}

/// What a store that never checkpointed holds, reopened, after `before`
/// and `seed`'s script replayed as `runs` say.
fn scripted_control(
    before: impl Fn(&Fx<MemoryBackend>),
    seed: u64,
    runs: &[(u32, bool)],
) -> (Contents, Vec<u32>) {
    let fx = Fx::memory();
    before(&fx);
    let script = Script::replayed(&fx, seed, runs);
    let reopened = Fx::over(&fx.log, fx.log.clone(), Options::default());
    scripted(&reopened, &script)
}

/// Half the records of the snapshot a checkpoint after `before` writes.
fn half_of(before: impl Fn(&Fx<MemoryBackend>)) -> u64 {
    let (_, records) = twin_snapshot(before).expect("a snapshot");
    u64::try_from(records.len() / 2).unwrap()
}

/// Has a flight begun on this thread open the first gate on reaching
/// `at`, then wait at it until the second gate opens.
fn park_this_thread_at(at: StorePark) -> (Arc<Gate>, Arc<Gate>) {
    let reached = Arc::new(Gate::default());
    let release = Arc::new(Gate::default());
    let hook = {
        let reached = Arc::clone(&reached);
        let release = Arc::clone(&release);
        move |point| {
            if point == at {
                reached.open();
                release.wait_within(WAIT);
            }
        }
    };
    STORE_CHECKPOINT_PARK.set(Some(Arc::new(hook)));
    (reached, release)
}

/// Polls `flight` until its worker opens `reached`.
fn poll_until<B: Backend + 'static>(fx: &Fx<B>, flight: &StoreCheckpoint, reached: &Gate) {
    let started = std::time::Instant::now();
    while !reached.wait_within(Duration::from_millis(1)) {
        flight.poll(&fx.store, STEP).unwrap();
        assert!(
            started.elapsed() < WAIT,
            "the worker never reached its park"
        );
    }
}

/// For each seed, a flight over a populated store with half the script
/// applied across polls and the rest while its worker waits at `at`,
/// given half the snapshot's records; the snapshot is a blocking
/// checkpoint's at begin and a reopen equals a store that never
/// checkpointed. The script declares nothing at the committed park, where a
/// declare waits for the verdict; the `reconcile_wait_*` tests cover that.
fn parked_with_edits(at: fn(u64) -> StorePark) {
    let before = |fx: &Fx<MemoryBackend>| fx.populate(SCRIPT_ROWS);
    let half = half_of(before);
    let twin = twin_snapshot(before);
    for seed in SEEDS {
        let fx = Fx::memory();
        before(&fx);
        let point = at(half);
        let (reached, release) = park_this_thread_at(point);
        let flight = fx.begin();
        let mut script = Script::new(seed);
        for _ in 0..EDITS / 2 {
            script.edit(&fx);
            flight.poll(&fx.store, STEP).unwrap();
        }
        poll_until(&fx, &flight, &reached);
        script.declares = declares_at(point);
        script.edits(&fx, EDITS - EDITS / 2);
        release.open();
        let upto = fx.finish(&flight);
        STORE_CHECKPOINT_PARK.set(None);
        assert!(!fx.store.capture_installed());
        let snapshot = fx.snapshot();
        assert_eq!(snapshot.as_ref().map(|(at, _)| *at), Some(upto));
        assert_eq!(snapshot, twin, "seed {seed}");
        let reopened = Fx::over(&fx.log, fx.log.clone(), Options::default());
        assert_eq!(
            scripted(&reopened, &script),
            scripted_control(
                before,
                seed,
                &[(EDITS / 2, true), (EDITS - EDITS / 2, declares_at(point))]
            ),
            "seed {seed}"
        );
    }
}

#[test]
fn park_edits_while_the_worker_waits_captured() {
    parked_with_edits(|_| StorePark::Captured);
}

#[test]
fn park_edits_while_the_worker_waits_half_pulled() {
    parked_with_edits(StorePark::Pulled);
}

#[test]
fn park_edits_while_the_worker_waits_committed() {
    parked_with_edits(|_| StorePark::Committed);
}

const CHILD_DIR: &str = "INCR_STORE_FLIGHT_CHILD_DIR";
const KILL_ROWS: u32 = 2_000;
const KILL_SEED: u64 = 7;

/// Kills the child process when dropped, so a failed wait leaves none.
struct Child(std::process::Child);

impl Drop for Child {
    fn drop(&mut self) {
        if let Err(error) = self.0.kill().and_then(|()| self.0.wait().map(drop)) {
            eprintln!("the child was not killed: {error}");
        }
    }
}

/// A store on a file log in `dir` whose writes skip `sync`: a killed
/// process still leaves them in the file.
fn file_fx(dir: &std::path::Path) -> Fx<FileBackend> {
    Fx::over(
        &MemoryBackend::new(),
        FileBackend::open(dir.join("log")).unwrap(),
        Options {
            durability: Durability::Buffered,
            ..Options::default()
        },
    )
}

/// The writes the child makes before its flight begins, a checkpoint
/// among them, which leave `script` to go on with.
fn before_kill<B: Backend + 'static>(fx: &Fx<B>, script: &mut Script) {
    fx.fill(KILL_ROWS);
    fx.store.checkpoint().unwrap();
    script.edits(fx, EDITS / 3);
}

/// The child: fills a store on a file log and checkpoints it, edits,
/// notes the snapshot id and the position, begins a flight whose worker
/// stops for good at `at`, edits until and while it is stopped there,
/// marks `parked` and waits to be killed.
fn child_parked_for_good(dir: &std::path::Path, at: fn(u64) -> StorePark) -> ! {
    let half = half_of(|fx| before_kill(fx, &mut Script::new(KILL_SEED)));
    let fx = file_fx(dir);
    let mut script = Script::new(KILL_SEED);
    before_kill(&fx, &mut script);
    let previous = FileBackend::open_read_only(dir.join("log"))
        .unwrap()
        .snapshot_id()
        .unwrap()
        .expect("the checkpoint's snapshot");
    let begun = fx.store.stats().unwrap().position;
    std::fs::write(
        dir.join("previous"),
        format!(
            "{} {} {}",
            previous.position.0, previous.generation, begun.0
        ),
    )
    .unwrap();
    let reached = Arc::new(Gate::default());
    let hook = {
        let reached = Arc::clone(&reached);
        move |point| {
            if point == at(half) {
                reached.open();
                loop {
                    std::thread::park();
                }
            }
        }
    };
    STORE_CHECKPOINT_PARK.set(Some(Arc::new(hook)));
    let flight = fx.begin();
    for _ in 0..EDITS / 3 {
        script.edit(&fx);
        flight.poll(&fx.store, STEP).unwrap();
    }
    poll_until(&fx, &flight, &reached);
    script.declares = declares_at(at(half));
    script.edits(&fx, EDITS - 2 * (EDITS / 3));
    std::fs::write(dir.join("parked"), b"").unwrap();
    loop {
        std::thread::park();
    }
}

/// Runs this test's child, kills it once its worker is stopped at `at`,
/// and reopens the file log: on the snapshot the child noted unless
/// `committed`, else on the flight's own snapshot at the noted position;
/// either way equal to a store that never checkpointed.
fn killed_at(name: &str, at: fn(u64) -> StorePark, committed: bool) {
    if let Ok(dir) = std::env::var(CHILD_DIR) {
        child_parked_for_good(std::path::Path::new(&dir), at);
    }
    let dir = tempfile::tempdir().unwrap();
    let test = module_path!().split_once("::").unwrap().1;
    let mut child = Child(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args([&format!("{test}::{name}"), "--exact", "--test-threads=1"])
            .env(CHILD_DIR, dir.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    let started = std::time::Instant::now();
    while !dir.path().join("parked").exists() {
        if let Some(status) = child.0.try_wait().unwrap() {
            panic!("the child exited before its park: {status}");
        }
        assert!(started.elapsed() < WAIT, "the child never reached its park");
        std::thread::sleep(Duration::from_millis(10));
    }
    child.0.kill().unwrap();
    child.0.wait().unwrap();
    let noted: Vec<u64> = std::fs::read_to_string(dir.path().join("previous"))
        .unwrap()
        .split(' ')
        .map(|n| n.parse().unwrap())
        .collect();
    let previous = SnapshotId {
        position: Position(noted[0]),
        generation: noted[1],
    };
    let expected = if committed {
        SnapshotId {
            position: Position(noted[2]),
            generation: previous.generation + 1,
        }
    } else {
        previous
    };
    let id = FileBackend::open_read_only(dir.path().join("log"))
        .unwrap()
        .snapshot_id()
        .unwrap();
    assert_eq!(id, Some(expected));
    let runs = [
        (2 * (EDITS / 3), true),
        (EDITS - 2 * (EDITS / 3), declares_at(at(0))),
    ];
    let script = Script::replayed(&Fx::memory(), KILL_SEED, &runs);
    assert_eq!(
        scripted(&file_fx(dir.path()), &script),
        scripted_control(|fx| fx.fill(KILL_ROWS), KILL_SEED, &runs)
    );
}

#[test]
fn kill_a_child_parked_captured_reopens_on_the_previous_snapshot() {
    killed_at(
        "kill_a_child_parked_captured_reopens_on_the_previous_snapshot",
        |_| StorePark::Captured,
        false,
    );
}

#[test]
fn kill_a_child_parked_half_pulled_reopens_on_the_previous_snapshot() {
    killed_at(
        "kill_a_child_parked_half_pulled_reopens_on_the_previous_snapshot",
        StorePark::Pulled,
        false,
    );
}

#[test]
fn kill_a_child_parked_committed_reopens_on_the_new_snapshot() {
    killed_at(
        "kill_a_child_parked_committed_reopens_on_the_new_snapshot",
        |_| StorePark::Committed,
        true,
    );
}
