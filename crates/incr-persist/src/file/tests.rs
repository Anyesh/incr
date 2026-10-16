use std::fs::{self, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::thread;
use std::time::Duration;

use proptest::prelude::*;

use super::format::{self, journal_path, snapshot_path, HEADER_LEN, JOURNAL_MAGIC, SNAPSHOT_MAGIC};
use super::FileBackend;
use crate::{Backend, Error, Kind, Position, ReadOutcome, Record};

type Contents = (Option<(Position, Vec<Record>)>, Vec<(Position, Record)>);

fn set(key: &str, value: u64) -> Record {
    Record::Set {
        key: key.into(),
        value: value.to_le_bytes().to_vec(),
    }
}

fn declare(key: &str) -> Record {
    Record::Declare {
        key: key.into(),
        kind: Kind::Input,
        default: vec![],
    }
}

fn read_all(backend: &FileBackend, after: Position) -> Vec<(Position, Record)> {
    let mut out = Vec::new();
    assert_eq!(
        backend.read_from(after, &mut out).unwrap(),
        ReadOutcome::Records
    );
    out
}

fn contents(backend: &FileBackend) -> Contents {
    let snapshot = backend.load_snapshot().unwrap();
    let from = snapshot.as_ref().map_or(Position::ZERO, |(at, _)| *at);
    (snapshot, read_all(backend, from))
}

fn files(dir: &Path) -> Vec<String> {
    let mut names: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

fn header_position(path: &Path, magic: [u8; 8]) -> u64 {
    let mut bytes = [0; HEADER_LEN];
    fs::File::open(path)
        .unwrap()
        .read_exact(&mut bytes)
        .unwrap();
    format::parse_header(&bytes, magic).unwrap()
}

#[test]
fn open_creates_lock_current_and_the_first_journal() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("state");
    let backend = FileBackend::open(&dir).unwrap();
    assert!(backend.writable());
    assert_eq!(files(&dir), ["CURRENT", "LOCK", "journal.0"]);
    assert_eq!(format::read_current(&dir).unwrap(), 0);
    assert_eq!(header_position(&journal_path(&dir, 0), JOURNAL_MAGIC), 1);
    assert_eq!(backend.load_snapshot().unwrap(), None);
}

#[test]
fn second_writable_open_fails_until_the_first_closes() {
    let dir = tempfile::tempdir().unwrap();
    let first = FileBackend::open(dir.path()).unwrap();
    let err = FileBackend::open(dir.path()).unwrap_err();
    match &err {
        Error::Io(e) => assert_eq!(e.kind(), io::ErrorKind::ResourceBusy),
        other => panic!("expected a busy error, got {other:?}"),
    }
    assert!(
        err.to_string().contains("already open for writing"),
        "{err}"
    );
    drop(first);
    FileBackend::open(dir.path()).unwrap();
}

#[test]
fn read_only_opens_take_no_lock_and_need_an_existing_log() {
    let dir = tempfile::tempdir().unwrap();
    let missing = FileBackend::open_read_only(dir.path().join("nothing")).unwrap_err();
    assert!(matches!(&missing, Error::Io(e) if e.kind() == io::ErrorKind::NotFound));

    let writer = FileBackend::open(dir.path()).unwrap();
    let first = FileBackend::open_read_only(dir.path()).unwrap();
    let second = FileBackend::open_read_only(dir.path()).unwrap();
    assert!(!first.writable());
    assert!(matches!(first.sync(Position(1)), Err(Error::ReadOnly)));
    let last = writer.append(&[declare("a"), set("a", 1)]).unwrap();
    assert_eq!(read_all(&first, Position::ZERO).len(), 2);
    drop(writer);
    assert_eq!(read_all(&second, Position::ZERO).last().unwrap().0, last);
    FileBackend::open(dir.path()).unwrap();
}

#[test]
fn compaction_moves_to_a_new_generation_and_removes_the_old_pair() {
    let dir = tempfile::tempdir().unwrap();
    let backend = FileBackend::open(dir.path()).unwrap();
    backend
        .append(&[declare("a"), set("a", 1), set("a", 2)])
        .unwrap();
    backend.compact(Position(2), vec![set("a", 1)]).unwrap();
    assert_eq!(
        files(dir.path()),
        ["CURRENT", "LOCK", "journal.1", "snapshot.1"]
    );
    assert_eq!(format::read_current(dir.path()).unwrap(), 1);
    assert_eq!(
        header_position(&snapshot_path(dir.path(), 1), SNAPSHOT_MAGIC),
        2
    );
    assert_eq!(
        header_position(&journal_path(dir.path(), 1), JOURNAL_MAGIC),
        3
    );
    assert_eq!(
        contents(&backend),
        (
            Some((Position(2), vec![set("a", 1)])),
            vec![(Position(3), set("a", 2))]
        )
    );
}

#[test]
fn unknown_file_format_is_rejected_by_every_open() {
    let dir = tempfile::tempdir().unwrap();
    drop(FileBackend::open(dir.path()).unwrap());
    let mut journal = OpenOptions::new()
        .write(true)
        .open(journal_path(dir.path(), 0))
        .unwrap();
    journal.seek(SeekFrom::Start(8)).unwrap();
    journal.write_all(&[format::FILE_FORMAT + 1]).unwrap();
    drop(journal);
    assert!(matches!(
        FileBackend::open(dir.path()),
        Err(Error::UnsupportedFormat { .. })
    ));
    assert!(matches!(
        FileBackend::open_read_only(dir.path()),
        Err(Error::UnsupportedFormat { .. })
    ));
}

fn batch_records(index: usize, sizes: &[usize]) -> Vec<Record> {
    sizes
        .iter()
        .map(|&size| Record::Set {
            key: format!("k{index}"),
            value: vec![index as u8; size],
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    // Damage lands at an arbitrary byte of the frame area, either cutting the
    // file there or flipping that byte. Every append that ended before the
    // damage must survive whole; nothing at or after it may appear.
    #[test]
    fn torn_tail_recovery_keeps_exactly_the_appends_before_the_damage(
        batches in prop::collection::vec(prop::collection::vec(0usize..300, 1..4), 1..10),
        damage in any::<prop::sample::Index>(),
        flip in prop::option::of(1u8..=255),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = journal_path(dir.path(), 0);
        let mut appended = Vec::new();
        {
            let writer = FileBackend::open(dir.path()).unwrap();
            for (index, sizes) in batches.iter().enumerate() {
                let records = batch_records(index, sizes);
                let last = writer.append(&records).unwrap();
                writer.sync(last).unwrap();
                let end = fs::metadata(&path).unwrap().len();
                appended.push((end, last, records));
            }
        }
        let len = fs::metadata(&path).unwrap().len();
        let offset = HEADER_LEN as u64 + damage.index((len - HEADER_LEN as u64) as usize) as u64;
        let mut file = OpenOptions::new().read(true).write(true).open(&path).unwrap();
        match flip {
            None => file.set_len(offset).unwrap(),
            Some(mask) => {
                let mut byte = [0];
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.read_exact(&mut byte).unwrap();
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.write_all(&[byte[0] ^ mask]).unwrap();
            }
        }
        drop(file);
        let damaged_len = fs::metadata(&path).unwrap().len();

        let mut want = Vec::new();
        let mut kept_end = HEADER_LEN as u64;
        for (end, last, records) in &appended {
            if *end > offset {
                break;
            }
            kept_end = *end;
            let first = last.0 + 1 - records.len() as u64;
            want.extend((first..).map(Position).zip(records.iter().cloned()));
        }

        let reader = FileBackend::open_read_only(dir.path()).unwrap();
        prop_assert_eq!(read_all(&reader, Position::ZERO), want.clone());
        prop_assert_eq!(fs::metadata(&path).unwrap().len(), damaged_len);
        drop(reader);

        let writer = FileBackend::open(dir.path()).unwrap();
        prop_assert_eq!(read_all(&writer, Position::ZERO), want.clone());
        prop_assert_eq!(fs::metadata(&path).unwrap().len(), kept_end);
        let next = writer.append(&[set("after", 1)]).unwrap();
        prop_assert_eq!(next, Position(want.len() as u64 + 1));
    }
}

// Simulates a crash after each of the five compaction steps by stopping the
// compaction there and dropping the handle. Steps 1-3 happen before CURRENT
// moves, so the old state must read back whole; steps 4-5 come after it, so
// the new state must.
#[test]
fn a_crash_after_any_compaction_step_leaves_the_old_or_new_state_whole() {
    for step in 1..=5u8 {
        let dir = tempfile::tempdir().unwrap();
        let upto = Position(4);
        let folded = vec![declare("a"), set("a", 3)];
        let before = {
            let writer = FileBackend::open(dir.path()).unwrap();
            writer
                .append(&[declare("a"), set("a", 1), set("a", 2)])
                .unwrap();
            writer
                .compact(Position(2), vec![declare("a"), set("a", 1)])
                .unwrap();
            writer.append(&[set("a", 3), set("a", 4)]).unwrap();
            let before = contents(&writer);
            writer.compact_steps(upto, folded.clone(), step).unwrap();
            before
        };
        let after = (
            Some((upto, folded.clone())),
            vec![(Position(5), set("a", 4))],
        );
        let (expected, generation) = if step >= 4 { (&after, 2) } else { (&before, 1) };

        let reader = FileBackend::open_read_only(dir.path()).unwrap();
        assert_eq!(
            &contents(&reader),
            expected,
            "read-only, crash after step {step}"
        );
        drop(reader);

        let writer = FileBackend::open(dir.path()).unwrap();
        assert_eq!(
            &contents(&writer),
            expected,
            "writer, crash after step {step}"
        );
        assert_eq!(
            files(dir.path()),
            [
                "CURRENT".to_string(),
                "LOCK".to_string(),
                format!("journal.{generation}"),
                format!("snapshot.{generation}"),
            ],
            "leftovers after step {step}"
        );

        let next = writer.append(&[set("a", 9)]).unwrap();
        assert_eq!(next, Position(6));
        writer.compact(next, vec![set("a", 9)]).unwrap();
        drop(writer);
        let writer = FileBackend::open(dir.path()).unwrap();
        assert_eq!(
            contents(&writer),
            (Some((next, vec![set("a", 9)])), vec![]),
            "compacting again after a crash at step {step}"
        );
    }
}

#[test]
fn reader_drains_its_unlinked_journal_then_continues_on_the_new_one() {
    let dir = tempfile::tempdir().unwrap();
    let writer = FileBackend::open(dir.path()).unwrap();
    writer.append(&[declare("a"), set("a", 1)]).unwrap();
    let reader = FileBackend::open_read_only(dir.path()).unwrap();
    assert_eq!(read_all(&reader, Position::ZERO).len(), 2);

    writer.append(&[set("a", 2), set("a", 3)]).unwrap();
    writer.compact(Position(4), vec![set("a", 3)]).unwrap();
    assert!(!journal_path(dir.path(), 0).exists());

    assert_eq!(
        read_all(&reader, Position(2)),
        vec![(Position(3), set("a", 2)), (Position(4), set("a", 3))]
    );
    assert!(read_all(&reader, Position(4)).is_empty());
    let next = writer.append(&[set("a", 4)]).unwrap();
    assert_eq!(read_all(&reader, Position(4)), vec![(next, set("a", 4))]);
}

#[test]
fn one_read_drains_the_old_journal_and_continues_into_the_new_one() {
    let dir = tempfile::tempdir().unwrap();
    let writer = FileBackend::open(dir.path()).unwrap();
    writer.append(&[declare("a"), set("a", 1)]).unwrap();
    let reader = FileBackend::open_read_only(dir.path()).unwrap();
    assert_eq!(read_all(&reader, Position::ZERO).len(), 2);

    writer.append(&[set("a", 2), set("a", 3)]).unwrap();
    writer.compact(Position(4), vec![set("a", 3)]).unwrap();
    writer.append(&[set("a", 4), set("a", 5)]).unwrap();
    assert_eq!(
        read_all(&reader, Position(2)),
        vec![
            (Position(3), set("a", 2)),
            (Position(4), set("a", 3)),
            (Position(5), set("a", 4)),
            (Position(6), set("a", 5)),
        ]
    );
}

#[test]
fn reader_whose_journal_ends_below_the_newest_snapshot_gets_compacted() {
    let dir = tempfile::tempdir().unwrap();
    let writer = FileBackend::open(dir.path()).unwrap();
    writer.append(&[declare("a"), set("a", 1)]).unwrap();
    let reader = FileBackend::open_read_only(dir.path()).unwrap();
    assert_eq!(read_all(&reader, Position::ZERO).len(), 2);

    writer.append(&[set("a", 2)]).unwrap();
    writer.compact(Position(3), vec![set("a", 2)]).unwrap();
    writer.append(&[set("a", 3)]).unwrap();
    writer.compact(Position(4), vec![set("a", 3)]).unwrap();

    // Position 3 is still in the reader's unlinked journal, but position 4
    // survives only in the snapshot, so the drained part is not returned.
    let mut out = vec![(Position(1), declare("kept"))];
    assert_eq!(
        reader.read_from(Position(2), &mut out).unwrap(),
        ReadOutcome::Compacted {
            snapshot_at: Position(4)
        }
    );
    assert_eq!(out, vec![(Position(1), declare("kept"))]);
    let mut out = Vec::new();
    assert_eq!(
        reader.read_from(Position(3), &mut out).unwrap(),
        ReadOutcome::Compacted {
            snapshot_at: Position(4)
        }
    );
    assert!(out.is_empty());
    assert_eq!(
        reader.load_snapshot().unwrap(),
        Some((Position(4), vec![set("a", 3)]))
    );
    let next = writer.append(&[set("a", 4)]).unwrap();
    assert_eq!(read_all(&reader, Position(4)), vec![(next, set("a", 4))]);
}

// The writer stores the position's own number at every position and folds
// each snapshot to the record at its position, so the reader can check any
// record or snapshot it sees against the position it claims. The reader
// pauses now and then so that it sometimes falls behind more than one
// compaction and takes the Compacted path, not only the draining one.
#[test]
fn reader_racing_compaction_sees_every_position_once_and_in_order() {
    const LAST: u64 = 3000;
    let dir = tempfile::tempdir().unwrap();
    let writer = FileBackend::open(dir.path()).unwrap();
    let reader = FileBackend::open_read_only(dir.path()).unwrap();
    thread::scope(|s| {
        let writing = s.spawn(|| {
            for position in 1..=LAST {
                assert_eq!(
                    writer.append(&[set("k", position)]).unwrap(),
                    Position(position)
                );
                if position.is_multiple_of(89) {
                    let upto = position - (position / 89 % 3);
                    writer
                        .compact(Position(upto), vec![set("k", upto)])
                        .unwrap();
                }
            }
        });

        let mut applied = 0;
        let mut reads = 0u64;
        loop {
            let finished = writing.is_finished();
            let before = applied;
            let mut out = Vec::new();
            match reader.read_from(Position(applied), &mut out).unwrap() {
                ReadOutcome::Records => {
                    for (position, record) in out {
                        assert_eq!(position.0, applied + 1, "gap or repeat after {applied}");
                        assert_eq!(record, set("k", position.0));
                        applied = position.0;
                    }
                }
                ReadOutcome::Compacted { snapshot_at } => {
                    assert!(
                        snapshot_at.0 > applied,
                        "Compacted at or above the snapshot"
                    );
                    let (at, folded) = reader.load_snapshot().unwrap().unwrap();
                    assert!(at >= snapshot_at);
                    assert_eq!(folded, vec![set("k", at.0)]);
                    applied = at.0;
                }
            }
            if finished && applied == before {
                assert_eq!(applied, LAST);
                break;
            }
            reads += 1;
            if reads.is_multiple_of(32) {
                thread::sleep(Duration::from_millis(60));
            }
        }
    });
}
