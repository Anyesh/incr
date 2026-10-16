//! The contract every [`Backend`] must meet, as runnable checks.
//!
//! Implement [`Harness`] for a backend and invoke
//! [`backend_conformance!`](crate::backend_conformance) in a test file to
//! get one `#[test]` per case. Each case is also a public function taking
//! the harness by value, for backends that need custom test wiring. Cases
//! panic on a contract violation.
//!
//! "Reopen" in a case means dropping every handle and opening the log
//! again, which is as close to a restart as a generic suite can get. Crash
//! durability (torn writes, faults mid-compaction) is backend-specific and
//! tested by each backend itself.

use std::collections::{HashMap, HashSet};
use std::thread;

use crate::{Backend, Error, Kind, Position, ReadOutcome, Record};

/// Creates logs and opens handles onto them for the suite.
pub trait Harness {
    type Backend: Backend;
    /// Identifies one log, such as a directory, a database path or a shared
    /// in-memory handle.
    type Log;

    /// Creates a new, empty log.
    fn new_log(&mut self) -> Self::Log;

    /// Opens a writable handle on `log`. The suite drops all writable
    /// handles before opening another when [`Harness::multi_writer`] is
    /// `false`.
    fn open(&mut self, log: &Self::Log) -> Self::Backend;

    /// Opens a read-only handle on `log`, or `None` if the backend has no
    /// read-only mode. Read-only handles may coexist with a writer.
    fn open_read_only(&mut self, log: &Self::Log) -> Option<Self::Backend> {
        let _ = log;
        None
    }

    /// Whether several writable handles may be open on one log at once.
    /// Cases that need concurrent writers are skipped when this is `false`.
    fn multi_writer(&self) -> bool;
}

/// Expands to one `#[test]` per conformance case, each building its harness
/// from `$harness`. Requires the `backend-tests` feature.
///
/// ```ignore
/// incr_persist::backend_conformance!(MyHarness::new());
/// ```
#[macro_export]
macro_rules! backend_conformance {
    ($harness:expr) => {
        $crate::backend_conformance!(@cases $harness;
            append_assigns_contiguous_increasing_positions,
            empty_append_returns_the_tail,
            read_from_returns_only_later_records,
            synced_records_survive_reopen,
            records_roundtrip_unchanged,
            writers_share_one_total_order,
            readers_never_skip_records_under_concurrent_appends,
            compaction_keeps_later_records_and_records_its_position,
            compacted_when_behind_snapshot_with_no_log_records,
            positions_are_never_reused_after_compaction,
            compaction_survives_reopen,
            compaction_rejects_positions_outside_snapshot_and_tail,
            forgotten_keys_leave_snapshot_and_log,
            read_only_handle_rejects_writes_and_follows_writer,
        );
    };
    (@cases $harness:expr; $($case:ident),+ $(,)?) => {
        $(
            #[test]
            fn $case() {
                $crate::conformance::$case($harness);
            }
        )+
    };
}

fn set(key: &str, value: u8) -> Record {
    Record::Set {
        key: key.into(),
        value: vec![value],
    }
}

fn declare(key: &str) -> Record {
    Record::Declare {
        key: key.into(),
        kind: Kind::Input,
        default: vec![0],
    }
}

fn read_all<B: Backend>(backend: &B, after: Position) -> Vec<(Position, Record)> {
    let mut out = Vec::new();
    let outcome = backend
        .read_from(after, &mut out)
        .expect("read_from failed");
    assert_eq!(
        outcome,
        ReadOutcome::Records,
        "read_from({after}) reported compaction unexpectedly"
    );
    out
}

fn assert_compacted<B: Backend>(backend: &B, after: Position, snapshot_at: Position) {
    let mut out = vec![(Position(u64::MAX), set("sentinel", 0))];
    let outcome = backend
        .read_from(after, &mut out)
        .expect("read_from failed");
    assert_eq!(
        outcome,
        ReadOutcome::Compacted { snapshot_at },
        "read_from({after}) must report the snapshot it fell behind"
    );
    assert_eq!(out.len(), 1, "a Compacted read must push nothing");
}

fn assert_strictly_increasing(read: &[(Position, Record)]) {
    for pair in read.windows(2) {
        assert!(
            pair[0].0 < pair[1].0,
            "positions out of order: {} then {}",
            pair[0].0,
            pair[1].0
        );
    }
}

fn positions_of(last: Position, len: usize) -> Vec<Position> {
    (0..len as u64)
        .map(|i| Position(last.0 - (len as u64 - 1) + i))
        .collect()
}

/// One append occupies consecutive positions ending at the returned one,
/// and the next append lands after it.
pub fn append_assigns_contiguous_increasing_positions<H: Harness>(mut h: H) {
    let log = h.new_log();
    let backend = h.open(&log);
    let first = vec![declare("a"), set("a", 1), set("a", 2)];
    let last = backend.append(&first).unwrap();
    assert!(last > Position::ZERO, "positions start above ZERO");

    let read = read_all(&backend, Position::ZERO);
    let expected: Vec<_> = positions_of(last, first.len())
        .into_iter()
        .zip(first)
        .collect();
    assert_eq!(read, expected, "one append must occupy a contiguous run");

    let next = backend.append(&[set("a", 3)]).unwrap();
    assert!(next > last, "a later append must land after an earlier one");
    assert_eq!(read_all(&backend, last), vec![(next, set("a", 3))]);
}

/// An empty append writes nothing and reports the current tail.
pub fn empty_append_returns_the_tail<H: Harness>(mut h: H) {
    let log = h.new_log();
    let backend = h.open(&log);
    assert_eq!(backend.append(&[]).unwrap(), Position::ZERO);
    let tail = backend.append(&[declare("a"), set("a", 1)]).unwrap();
    assert_eq!(backend.append(&[]).unwrap(), tail);
    assert_eq!(read_all(&backend, Position::ZERO).len(), 2);
}

/// `read_from(after)` pushes exactly the records above `after`, appending
/// to `out` rather than replacing it.
pub fn read_from_returns_only_later_records<H: Harness>(mut h: H) {
    let log = h.new_log();
    let backend = h.open(&log);
    let mut positions = Vec::new();
    for value in 0..5 {
        positions.push(backend.append(&[set("a", value)]).unwrap());
    }

    let mut out = vec![(Position(u64::MAX), set("kept", 9))];
    assert_eq!(
        backend.read_from(positions[1], &mut out).unwrap(),
        ReadOutcome::Records
    );
    assert_eq!(out[0], (Position(u64::MAX), set("kept", 9)));
    let expected: Vec<_> = (2..5).map(|i| (positions[i], set("a", i as u8))).collect();
    assert_eq!(&out[1..], &expected[..]);

    assert!(read_all(&backend, positions[4]).is_empty());
    assert!(read_all(&backend, Position(positions[4].0 + 1000)).is_empty());
}

/// After `sync`, a reopened log returns the records.
pub fn synced_records_survive_reopen<H: Harness>(mut h: H) {
    let log = h.new_log();
    let records = vec![declare("a"), set("a", 7)];
    let last = {
        let backend = h.open(&log);
        let last = backend.append(&records).unwrap();
        backend.sync(last).unwrap();
        let more = backend.append(&[set("a", 8)]).unwrap();
        backend.sync(Position(more.0 + 100)).unwrap();
        more
    };
    let backend = h.open(&log);
    let read = read_all(&backend, Position::ZERO);
    assert_eq!(read.len(), 3);
    assert_eq!(read[2], (last, set("a", 8)));
    let contents: Vec<_> = read.into_iter().map(|(_, r)| r).take(2).collect();
    assert_eq!(contents, records);
}

/// Every record variant comes back byte for byte, from the log and from a
/// snapshot, including empty and large payloads and non-ASCII keys.
pub fn records_roundtrip_unchanged<H: Harness>(mut h: H) {
    let records = vec![
        Record::Declare {
            key: "input".into(),
            kind: Kind::Input,
            default: vec![],
        },
        Record::Declare {
            key: "coll".into(),
            kind: Kind::Collection,
            default: vec![],
        },
        Record::Set {
            key: "input".into(),
            value: (0..=255).cycle().take(70_000).collect(),
        },
        Record::Insert {
            key: "coll".into(),
            element: vec![1, 2],
        },
        Record::Remove {
            key: "coll".into(),
            element: vec![1, 2],
        },
        Record::Replace {
            key: "coll".into(),
            old: None,
            new: vec![3],
        },
        Record::Replace {
            key: "coll".into(),
            old: Some(vec![3]),
            new: vec![],
        },
        Record::Delete {
            key: "naïve/キー/\u{0}".into(),
        },
    ];
    let log = h.new_log();
    {
        let backend = h.open(&log);
        let last = backend.append(&records).unwrap();
        backend.sync(last).unwrap();
        let read: Vec<_> = read_all(&backend, Position::ZERO)
            .into_iter()
            .map(|(_, r)| r)
            .collect();
        assert_eq!(read, records);
        backend.compact(last, records.clone()).unwrap();
    }
    let backend = h.open(&log);
    let (_, snapshot) = backend.load_snapshot().unwrap().expect("snapshot");
    assert_eq!(snapshot, records);
}

const WRITERS: u8 = 4;
const APPENDS_PER_WRITER: u8 = 100;

fn writer_batch(writer: u8, seq: u8) -> Vec<Record> {
    let key = format!("w{writer}");
    (0..=(seq % 3))
        .map(|part| Record::Replace {
            key: key.clone(),
            old: Some(vec![seq]),
            new: vec![part],
        })
        .collect()
}

/// Concurrent writers on separate handles produce one total order: each
/// append is contiguous at the positions it returned, every record appears
/// exactly once, each writer's appends keep their order, and two handles
/// read identical logs. Skipped for single-writer backends.
pub fn writers_share_one_total_order<H: Harness>(mut h: H) {
    if !h.multi_writer() {
        return;
    }
    let log = h.new_log();
    let handles: Vec<_> = (0..WRITERS).map(|_| h.open(&log)).collect();
    let returned: Vec<Vec<(Position, Vec<Record>)>> = thread::scope(|s| {
        let joins: Vec<_> = handles
            .iter()
            .enumerate()
            .map(|(writer, backend)| {
                s.spawn(move || {
                    (0..APPENDS_PER_WRITER)
                        .map(|seq| {
                            let batch = writer_batch(writer as u8, seq);
                            (backend.append(&batch).unwrap(), batch)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        joins.into_iter().map(|j| j.join().unwrap()).collect()
    });

    let read = read_all(&handles[0], Position::ZERO);
    assert_strictly_increasing(&read);
    let by_position: HashMap<Position, &Record> = read.iter().map(|(p, r)| (*p, r)).collect();
    let mut expected_len = 0;
    for appends in &returned {
        let mut previous = Position::ZERO;
        for (last, batch) in appends {
            assert!(*last > previous, "a writer's appends must keep their order");
            previous = *last;
            expected_len += batch.len();
            for (position, record) in positions_of(*last, batch.len()).iter().zip(batch) {
                assert_eq!(
                    by_position.get(position).copied(),
                    Some(record),
                    "append returning {last} is not contiguous at {position}"
                );
            }
        }
    }
    assert_eq!(read.len(), expected_len, "records lost or duplicated");

    let other = h.open(&log);
    assert_eq!(read_all(&other, Position::ZERO), read);
}

/// A reader polling `read_from` from its last seen position while writers
/// append never skips a record: what it accumulates equals the final log.
pub fn readers_never_skip_records_under_concurrent_appends<H: Harness>(mut h: H) {
    let log = h.new_log();
    let writers: Vec<_> = if h.multi_writer() {
        (0..WRITERS).map(|_| h.open(&log)).collect()
    } else {
        vec![h.open(&log)]
    };
    let reader = match h.open_read_only(&log) {
        Some(reader) => reader,
        None if h.multi_writer() => h.open(&log),
        None => return,
    };
    let seen = thread::scope(|s| {
        let joins: Vec<_> = writers
            .iter()
            .enumerate()
            .map(|(writer, backend)| {
                s.spawn(move || {
                    for seq in 0..APPENDS_PER_WRITER {
                        backend.append(&writer_batch(writer as u8, seq)).unwrap();
                    }
                })
            })
            .collect();
        let mut seen = Vec::new();
        let mut applied = Position::ZERO;
        loop {
            let done = joins.iter().all(|j| j.is_finished());
            let start = seen.len();
            assert_eq!(
                reader.read_from(applied, &mut seen).unwrap(),
                ReadOutcome::Records
            );
            if let Some((first, _)) = seen.get(start) {
                assert!(
                    *first > applied,
                    "read_from({applied}) returned position {first}"
                );
            }
            if let Some((last, _)) = seen.last() {
                applied = *last;
            }
            if done && seen.len() == start {
                break;
            }
            thread::yield_now();
        }
        seen
    });
    let final_log = read_all(&writers[0], Position::ZERO);
    assert_strictly_increasing(&seen);
    assert_eq!(seen.len(), final_log.len(), "reader skipped records");
    assert_eq!(seen, final_log);
}

/// Compacting through a middle position keeps the later records, records
/// the snapshot at that position, and reports `Compacted` to readers below
/// it without pushing anything.
pub fn compaction_keeps_later_records_and_records_its_position<H: Harness>(mut h: H) {
    let log = h.new_log();
    let backend = h.open(&log);
    let positions: Vec<_> = (0..5)
        .map(|v| backend.append(&[set("a", v)]).unwrap())
        .collect();
    let folded = vec![declare("a"), set("a", 2)];
    backend.compact(positions[2], folded.clone()).unwrap();

    assert_eq!(
        backend.load_snapshot().unwrap(),
        Some((positions[2], folded))
    );
    assert_eq!(
        read_all(&backend, positions[2]),
        vec![(positions[3], set("a", 3)), (positions[4], set("a", 4))]
    );
    assert_compacted(&backend, Position::ZERO, positions[2]);
    assert_compacted(&backend, positions[1], positions[2]);
}

/// Compacting through the tail leaves no log records, yet a reader below
/// the snapshot still gets `Compacted` and one at it gets nothing.
pub fn compacted_when_behind_snapshot_with_no_log_records<H: Harness>(mut h: H) {
    let log = h.new_log();
    let backend = h.open(&log);
    let first = backend.append(&[declare("a")]).unwrap();
    let tail = backend.append(&[set("a", 1), set("a", 2)]).unwrap();
    backend.compact(tail, vec![]).unwrap();

    assert_eq!(backend.load_snapshot().unwrap(), Some((tail, vec![])));
    assert_compacted(&backend, Position::ZERO, tail);
    assert_compacted(&backend, first, tail);
    assert!(read_all(&backend, tail).is_empty());
}

/// Appends after a compaction that emptied the log land above every
/// position handed out before, in the same session and after a reopen.
pub fn positions_are_never_reused_after_compaction<H: Harness>(mut h: H) {
    let log = h.new_log();
    let tail = {
        let backend = h.open(&log);
        backend.append(&[declare("a"), set("a", 1)]).unwrap();
        let tail = backend.append(&[set("a", 2)]).unwrap();
        backend.compact(tail, vec![set("a", 2)]).unwrap();
        let next = backend.append(&[set("a", 3)]).unwrap();
        assert!(next > tail, "position {next} reused after compaction");
        backend.sync(next).unwrap();
        backend.compact(next, vec![set("a", 3)]).unwrap();
        next
    };
    let backend = h.open(&log);
    assert_eq!(backend.append(&[]).unwrap(), tail);
    let next = backend.append(&[set("a", 4)]).unwrap();
    assert!(next > tail, "position {next} reused after reopen");
}

/// The snapshot and the records after it are what a reopen sees.
pub fn compaction_survives_reopen<H: Harness>(mut h: H) {
    let log = h.new_log();
    let (snapshot_at, later) = {
        let backend = h.open(&log);
        let snapshot_at = backend.append(&[declare("a"), set("a", 1)]).unwrap();
        let later = backend.append(&[set("a", 2)]).unwrap();
        backend
            .compact(snapshot_at, vec![declare("a"), set("a", 1)])
            .unwrap();
        backend.sync(later).unwrap();
        (snapshot_at, later)
    };
    let backend = h.open(&log);
    assert_eq!(
        backend.load_snapshot().unwrap(),
        Some((snapshot_at, vec![declare("a"), set("a", 1)]))
    );
    assert_compacted(&backend, Position::ZERO, snapshot_at);
    assert_eq!(read_all(&backend, snapshot_at), vec![(later, set("a", 2))]);
}

/// A log that was never compacted has no snapshot; `compact` refuses
/// positions beyond the tail or below the current snapshot, and accepts the
/// current snapshot position to replace the snapshot.
pub fn compaction_rejects_positions_outside_snapshot_and_tail<H: Harness>(mut h: H) {
    let log = h.new_log();
    let backend = h.open(&log);
    assert_eq!(backend.load_snapshot().unwrap(), None);
    let first = backend.append(&[declare("a")]).unwrap();
    let tail = backend.append(&[set("a", 1)]).unwrap();

    let beyond = Position(tail.0 + 1);
    assert!(matches!(
        backend.compact(beyond, vec![]),
        Err(Error::InvalidCompaction { upto, .. }) if upto == beyond
    ));
    assert_eq!(backend.load_snapshot().unwrap(), None);
    assert_eq!(read_all(&backend, Position::ZERO).len(), 2);

    backend.compact(tail, vec![set("a", 1)]).unwrap();
    assert!(matches!(
        backend.compact(first, vec![]),
        Err(Error::InvalidCompaction { upto, snapshot_at, .. })
            if upto == first && snapshot_at == tail
    ));
    assert_eq!(
        backend.load_snapshot().unwrap(),
        Some((tail, vec![set("a", 1)]))
    );

    backend.compact(tail, vec![set("a", 9)]).unwrap();
    assert_eq!(
        backend.load_snapshot().unwrap(),
        Some((tail, vec![set("a", 9)]))
    );
}

/// A fold that omits a key removes it: the snapshot does not mention it,
/// no log record for it survives, and declaring it again starts fresh.
pub fn forgotten_keys_leave_snapshot_and_log<H: Harness>(mut h: H) {
    let log = h.new_log();
    let backend = h.open(&log);
    backend
        .append(&[
            declare("keep"),
            set("keep", 1),
            declare("gone"),
            set("gone", 5),
        ])
        .unwrap();
    let upto = backend
        .append(&[Record::Delete { key: "gone".into() }])
        .unwrap();
    backend
        .compact(upto, vec![declare("keep"), set("keep", 1)])
        .unwrap();

    let (at, snapshot) = backend.load_snapshot().unwrap().expect("snapshot");
    assert_eq!(at, upto);
    let keys: HashSet<&str> = snapshot.iter().map(Record::key).collect();
    assert_eq!(keys, HashSet::from(["keep"]));
    assert!(read_all(&backend, upto).is_empty());

    let redeclared = backend.append(&[declare("gone")]).unwrap();
    assert!(redeclared > upto);
    assert_eq!(
        read_all(&backend, upto),
        vec![(redeclared, declare("gone"))]
    );
}

/// A read-only handle reports `writable() == false`, rejects `append` and
/// `compact` with [`Error::ReadOnly`], and sees what a writer appends and
/// compacts. Skipped for backends without a read-only mode.
pub fn read_only_handle_rejects_writes_and_follows_writer<H: Harness>(mut h: H) {
    let log = h.new_log();
    let writer = h.open(&log);
    let Some(reader) = h.open_read_only(&log) else {
        return;
    };
    assert!(writer.writable());
    assert!(!reader.writable());

    let first = writer.append(&[declare("a"), set("a", 1)]).unwrap();
    assert!(matches!(
        reader.append(&[set("a", 2)]),
        Err(Error::ReadOnly)
    ));
    assert!(matches!(reader.append(&[]), Err(Error::ReadOnly)));
    assert!(matches!(
        reader.compact(first, vec![]),
        Err(Error::ReadOnly)
    ));
    assert_eq!(read_all(&reader, Position::ZERO).len(), 2);

    let second = writer.append(&[set("a", 3)]).unwrap();
    assert_eq!(read_all(&reader, first), vec![(second, set("a", 3))]);

    writer.compact(second, vec![set("a", 3)]).unwrap();
    assert_compacted(&reader, first, second);
    assert_eq!(
        reader.load_snapshot().unwrap(),
        Some((second, vec![set("a", 3)]))
    );
}
