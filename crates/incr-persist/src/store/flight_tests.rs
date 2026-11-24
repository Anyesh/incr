use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize, Serializer};

use super::{Input, Options, PersistedCollection, PersistedKeyed, Store, StoreCheckpoint};
use crate::{
    Backend, ConditionalAppend, Error, Kind, Local, MemoryBackend, Position, ReadOutcome, Record,
    Result, SnapshotId,
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
        let open = self.open.lock().unwrap();
        drop(
            self.changed
                .wait_timeout_while(open, GATE_TIMEOUT, |open| !*open)
                .unwrap(),
        );
    }
}

fn on_worker() -> bool {
    std::thread::current().name() == Some("incr-checkpoint")
}

/// A [`MemoryBackend`] whose compaction may wait at a gate, on a checkpoint
/// worker only, or pull as a test says, and which may decline concurrent
/// compaction.
struct Wrap {
    inner: MemoryBackend,
    concurrent: bool,
    gate: Option<Arc<Gate>>,
    pull: Option<Box<Pull>>,
}

impl Wrap {
    fn new(inner: &MemoryBackend) -> Self {
        Wrap {
            inner: inner.clone(),
            concurrent: true,
            gate: None,
            pull: None,
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
        self.inner.read_from(after, out)
    }

    fn load_snapshot(&self) -> Result<Option<(Position, Vec<Record>)>> {
        self.inner.load_snapshot()
    }

    fn snapshot_id(&self) -> Result<Option<SnapshotId>> {
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
    let fx = Fx::over(log, log.clone(), Options::default());
    let mut keys = fx.store.restored_keys("").unwrap();
    keys.retain(|key| key != "idle");
    let ghost = fx.store.collection::<String>("ghost").unwrap();
    let mut ghost = ghost.handle().snapshot_values();
    ghost.sort();
    let mut contents = contents(&fx);
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
    let other = Fx::over(&fx.log, fx.log.clone(), Options::default());
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
