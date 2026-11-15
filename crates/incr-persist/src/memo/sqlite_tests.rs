use std::sync::mpsc;
use std::thread;

use super::*;
use crate::memo::{
    decode_entry, write_checkpoint, CheckpointInput, Entry, StoredDep, TableDesc, DEFAULT_MAX_BYTES,
};
use crate::SqliteBackend;

const BUILD: u128 = 0xb111d;

fn desc() -> TableDesc {
    TableDesc {
        name: "t".into(),
        version: 1,
        key_type: "u8".into(),
        value_type: "u8".into(),
    }
}

fn header(seq: u64) -> GenHeader {
    GenHeader {
        build: BUILD,
        seq,
        tables: vec![desc()],
    }
}

fn entry(key: u8) -> Entry {
    Entry {
        table: 0,
        key: vec![key],
        value: vec![key.wrapping_mul(3); 5],
        deps: vec![StoredDep::Input {
            key: format!("in{key}"),
            fp: u128::from(key),
        }],
        last_used: 0,
    }
}

/// Frames tagged for `header(1)`, which the tests that decode them write.
fn encoded(count: u8) -> Vec<EncodedEntry> {
    let tag = header(1).frame_tag();
    (0..count)
        .map(|key| EncodedEntry::new(&entry(key), tag).unwrap())
        .collect()
}

fn checkpoint(memo: &SqliteMemo, fresh: Vec<Entry>) -> crate::memo::Checkpointed {
    write_checkpoint(
        memo,
        CheckpointInput {
            build: BUILD,
            tables: vec![desc()],
            fresh,
            max_bytes: DEFAULT_MAX_BYTES,
        },
    )
    .unwrap()
}

struct Db {
    dir: tempfile::TempDir,
    path: PathBuf,
}

fn db() -> Db {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memo.sqlite");
    Db { dir, path }
}

fn raw(db: &Db) -> Connection {
    Connection::open(&db.path).unwrap()
}

fn frame_of(conn: &Connection, loc: u64) -> Vec<u8> {
    conn.query_row(
        "SELECT frame FROM memo_entry WHERE loc = ?1",
        [loc as i64],
        |row| row.get(0),
    )
    .unwrap()
}

fn set_frame(conn: &Connection, loc: u64, frame: &[u8]) {
    conn.execute(
        "UPDATE memo_entry SET frame = ?1 WHERE loc = ?2",
        params![frame, loc as i64],
    )
    .unwrap();
}

#[test]
fn flipped_frame_bytes_read_back_as_stored_and_fail_to_decode() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    let generation = memo
        .write_generation(None, GenId(1), &header(1), &encoded(1))
        .unwrap();
    let item = generation.items()[0];
    let conn = raw(&db);
    let good = frame_of(&conn, item.loc);
    for at in 0..good.len() {
        let mut bad = good.clone();
        bad[at] ^= 0x01;
        set_frame(&conn, item.loc, &bad);
        let read = memo.read_entry(generation.id(), &item).unwrap().unwrap();
        assert_eq!(read, bad);
        let tag = generation.frame_tag();
        assert!(decode_entry(&read, tag).is_err(), "flip at {at} decoded");
    }
}

#[test]
fn damaged_frames_are_counted_corrupt_and_dropped_at_checkpoint() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    let first = checkpoint(&memo, (0..4).map(entry).collect()).generation;
    let items = first.items().to_vec();
    let conn = raw(&db);
    let flipped = frame_of(&conn, items[0].loc);
    let mut bad = flipped.clone();
    *bad.last_mut().unwrap() ^= 0x80;
    set_frame(&conn, items[0].loc, &bad);
    let truncated = frame_of(&conn, items[1].loc);
    set_frame(&conn, items[1].loc, &truncated[..truncated.len() - 1]);

    let second = checkpoint(&memo, vec![entry(10), entry(11)]);
    assert!(!second.stats.delta);
    assert_eq!(second.stats.corrupt, 2);
    assert_eq!(second.stats.merged, 2);
    assert_eq!(second.generation.items().len(), 4);
    assert_eq!(second.superseded, [first.id()]);
}

#[test]
fn damaged_header_loads_an_empty_generation_that_the_next_checkpoint_replaces() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    let first = checkpoint(&memo, (0..3).map(entry).collect()).generation;
    let conn = raw(&db);
    let mut blob: Vec<u8> = conn
        .query_row("SELECT header FROM memo_generation", [], |row| row.get(0))
        .unwrap();
    let last = blob.len() - 1;
    blob[last] ^= 0x01;
    conn.execute("UPDATE memo_generation SET header = ?1", [&blob])
        .unwrap();

    let loaded = memo.load().unwrap().unwrap();
    assert_eq!(loaded.id(), first.id());
    assert!(loaded.items().is_empty());
    assert_eq!(loaded.header().build, 0);

    let second = checkpoint(&memo, vec![entry(9)]);
    assert_eq!(second.superseded, [first.id()]);
    assert_eq!(second.stats.merged, 0);
    assert_eq!(memo.load().unwrap().unwrap().items().len(), 1);
}

#[test]
fn damaged_index_rows_load_an_empty_generation() {
    for damage in [
        "UPDATE memo_entry SET len = -1",
        "UPDATE memo_entry SET tbl = 4294967296",
        "UPDATE memo_entry SET len = 'x'",
    ] {
        let db = db();
        let memo = SqliteMemo::open(&db.path).unwrap();
        memo.write_generation(None, GenId(1), &header(1), &encoded(2))
            .unwrap();
        raw(&db).execute(damage, []).unwrap();
        let loaded = memo.load().unwrap().unwrap();
        assert_eq!(loaded.id(), GenId(1), "{damage}");
        assert!(loaded.items().is_empty(), "{damage}");
    }
}

#[test]
fn swapped_generation_headers_load_an_empty_generation() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    memo.write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    memo.write_generation(Some(GenId(1)), GenId(2), &header(2), &encoded(3))
        .unwrap();
    raw(&db)
        .execute_batch(
            "UPDATE memo_generation SET id = -id;
             UPDATE memo_generation SET id = 3 + id;",
        )
        .unwrap();

    let loaded = memo.load().unwrap().unwrap();
    assert_eq!(loaded.id(), GenId(2));
    assert!(loaded.items().is_empty());
}

#[test]
fn current_pointed_back_at_an_older_generation_loads_that_one_whole() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    let old = encoded(2);
    memo.write_generation(None, GenId(1), &header(1), &old)
        .unwrap();
    memo.write_generation(Some(GenId(1)), GenId(2), &header(2), &encoded(3))
        .unwrap();
    raw(&db)
        .execute("UPDATE memo_current SET generation = 1", [])
        .unwrap();

    let loaded = memo.load().unwrap().unwrap();
    assert_eq!(loaded.id(), GenId(1));
    assert_eq!(loaded.header(), &header(1));
    assert_eq!(loaded.items().len(), old.len());
    for written in &old {
        let item = loaded.find(written.table, written.key_hash)[0];
        let frame = memo.read_entry(loaded.id(), &item).unwrap().unwrap();
        assert_eq!(frame, written.frame);
    }
}

#[test]
fn a_checkpoint_after_current_pointed_back_replaces_the_newer_generation() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    memo.write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    memo.write_generation(Some(GenId(1)), GenId(2), &header(2), &encoded(3))
        .unwrap();
    raw(&db)
        .execute("UPDATE memo_current SET generation = 1", [])
        .unwrap();

    let first = checkpoint(&memo, vec![entry(9)]);
    assert_eq!(first.superseded, [GenId(1)]);
    assert_eq!(memo.load().unwrap().unwrap().id(), first.generation.id());
    memo.remove_generation(GenId(1)).unwrap();
    let second = checkpoint(&memo, vec![entry(8)]);
    let loaded = memo.load().unwrap().unwrap();
    assert_eq!(loaded.id(), second.generation.id());
    assert_eq!(loaded.items().len(), 4);
}

#[test]
fn a_checkpoint_after_current_was_deleted_replaces_every_generation() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    memo.write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    memo.write_generation(Some(GenId(1)), GenId(2), &header(2), &encoded(3))
        .unwrap();
    raw(&db).execute("DELETE FROM memo_current", []).unwrap();
    assert!(memo.load().unwrap().is_none());

    let first = checkpoint(&memo, vec![entry(9)]);
    assert!(first.superseded.is_empty());
    let loaded = memo.load().unwrap().unwrap();
    assert_eq!(loaded.id(), first.generation.id());
    assert_eq!(loaded.items().len(), 1);
}

/// A buffered commit that never reached the disk is a write-ahead log tail
/// that is gone after a crash. The copy below keeps the main file only,
/// which is what survives when the log after the last checkpoint is lost.
#[test]
fn buffered_loss_rolls_back_to_an_earlier_whole_generation() {
    let db = db();
    let options = SqliteMemoOptions {
        durability: SqliteDurability::Buffered,
        ..SqliteMemoOptions::default()
    };
    let memo = SqliteMemo::open_with(&db.path, options).unwrap();
    let kept = encoded(3);
    memo.write_generation(None, GenId(1), &header(1), &kept)
        .unwrap();
    lock(&memo.writer)
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
        .unwrap();
    memo.write_generation(Some(GenId(1)), GenId(2), &header(2), &encoded(5))
        .unwrap();
    memo.remove_generation(GenId(1)).unwrap();

    let copy = db.dir.path().join("crashed.sqlite");
    std::fs::copy(&db.path, &copy).unwrap();
    let after = SqliteMemo::open(&copy).unwrap();
    let loaded = after.load().unwrap().unwrap();
    assert_eq!(loaded.id(), GenId(1));
    assert_eq!(loaded.header(), &header(1));
    for written in &kept {
        let item = loaded.find(written.table, written.key_hash)[0];
        let frame = after.read_entry(loaded.id(), &item).unwrap().unwrap();
        assert_eq!(frame, written.frame);
    }
}

#[test]
fn durability_and_busy_timeout_reach_the_writer() {
    let db = db();
    for (durability, synchronous) in [(SqliteDurability::Full, 2), (SqliteDurability::Buffered, 1)]
    {
        let memo = SqliteMemo::open_with(
            &db.path,
            SqliteMemoOptions {
                durability,
                busy_timeout: Duration::from_millis(1234),
            },
        )
        .unwrap();
        let writer = lock(&memo.writer);
        let found: i64 = writer
            .pragma_query_value(None, "synchronous", |row| row.get(0))
            .unwrap();
        assert_eq!(found, synchronous);
        let timeout: i64 = writer
            .pragma_query_value(None, "busy_timeout", |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 1234);
    }
}

#[test]
fn a_write_blocked_past_the_busy_timeout_is_busy() {
    let db = db();
    let memo = SqliteMemo::open_with(
        &db.path,
        SqliteMemoOptions {
            busy_timeout: Duration::from_millis(50),
            ..SqliteMemoOptions::default()
        },
    )
    .unwrap();
    let mut other = raw(&db);
    let held = other
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .unwrap();
    let written = memo.write_generation(None, GenId(1), &header(1), &encoded(1));
    assert!(matches!(written, Err(Error::Busy)), "got {written:?}");
    drop(held);
    assert!(memo.load().unwrap().is_none());
}

/// Restores keep reading the current generation while a checkpoint writes
/// the next one, through this handle or another connection.
#[test]
fn reads_do_not_wait_for_a_write_in_progress() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    let written = encoded(2);
    let generation = memo
        .write_generation(None, GenId(1), &header(1), &written)
        .unwrap();
    let mut other = raw(&db);
    let held = other
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .unwrap();
    held.execute("DELETE FROM memo_entry", []).unwrap();
    let writing = lock(&memo.writer);

    let (send, receive) = mpsc::channel();
    thread::scope(|scope| {
        scope.spawn(|| {
            let loaded = memo.load().unwrap().unwrap();
            let item = loaded.items()[0];
            let frame = memo.read_entry(generation.id(), &item).unwrap();
            send.send((loaded.items().len(), frame.is_some())).unwrap();
        });
        let got = receive.recv_timeout(Duration::from_secs(10));
        drop(writing);
        assert_eq!(got, Ok((written.len(), true)));
    });
    drop(held);
}

#[test]
fn removing_the_current_generation_fails_and_keeps_it() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    memo.write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    assert!(memo.remove_generation(GenId(1)).is_err());
    assert_eq!(memo.load().unwrap().unwrap().items().len(), 2);
}

#[test]
fn a_generation_id_that_is_not_newer_is_refused() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    memo.write_generation(None, GenId(5), &header(1), &encoded(1))
        .unwrap();
    let reused = memo.write_generation(Some(GenId(5)), GenId(5), &header(2), &encoded(1));
    assert!(matches!(reused, Err(Error::Backend(_))), "got {reused:?}");
    assert_eq!(memo.load().unwrap().unwrap().id(), GenId(5));
}

#[test]
fn a_database_of_another_kind_is_rejected() {
    let db = db();
    drop(SqliteBackend::open(&db.path).unwrap());
    assert!(matches!(SqliteMemo::open(&db.path), Err(Error::Backend(_))));

    let other = db.dir.path().join("newer.sqlite");
    drop(SqliteMemo::open(&other).unwrap());
    Connection::open(&other)
        .unwrap()
        .pragma_update(None, "user_version", DELTA_SCHEMA_VERSION + 1)
        .unwrap();
    assert!(matches!(SqliteMemo::open(&other), Err(Error::Backend(_))));
}

#[test]
fn another_applications_database_is_rejected_and_left_unchanged() {
    let db = db();
    raw(&db)
        .execute_batch("CREATE TABLE users (id INTEGER)")
        .unwrap();
    assert!(matches!(SqliteMemo::open(&db.path), Err(Error::Backend(_))));

    let conn = raw(&db);
    let mode: String = conn
        .pragma_query_value(None, "journal_mode", |row| row.get(0))
        .unwrap();
    assert_eq!(mode, "delete");
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(tables, ["users"]);
    let application_id: i32 = conn
        .pragma_query_value(None, "application_id", |row| row.get(0))
        .unwrap();
    assert_eq!(application_id, 0);
}

#[test]
fn a_rejected_database_stays_in_its_journal_mode() {
    let db = db();
    raw(&db).pragma_update(None, "user_version", 7).unwrap();
    assert!(matches!(SqliteMemo::open(&db.path), Err(Error::Backend(_))));
    let mode: String = raw(&db)
        .pragma_query_value(None, "journal_mode", |row| row.get(0))
        .unwrap();
    assert_eq!(mode, "delete");
}

#[test]
fn concurrent_first_opens_create_one_schema() {
    const OPENERS: usize = 8;
    let db = db();
    let barrier = std::sync::Barrier::new(OPENERS);
    thread::scope(|scope| {
        for _ in 0..OPENERS {
            scope.spawn(|| {
                barrier.wait();
                let memo = SqliteMemo::open_with(
                    &db.path,
                    SqliteMemoOptions {
                        busy_timeout: Duration::from_secs(60),
                        ..SqliteMemoOptions::default()
                    },
                )
                .unwrap();
                memo.load().unwrap();
            });
        }
    });
}

fn user_version(conn: &Connection) -> i64 {
    conn.pragma_query_value(None, "user_version", |row| row.get(0))
        .unwrap()
}

fn has_parent_column(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT count(*) FROM pragma_table_info('memo_generation') WHERE name = 'parent'",
        [],
        |row| row.get::<_, i64>(0),
    )
    .unwrap()
        > 0
}

fn delta_header(base: u64, parent: u64, id: u64) -> DeltaHeader {
    DeltaHeader {
        header: header(id),
        base: GenId(base),
        parent: GenId(parent),
    }
}

fn delta_entry(key: u8, delta: &DeltaHeader) -> EncodedEntry {
    EncodedEntry::new(&entry(key), delta.frame_tag()).unwrap()
}

fn write_delta_on(
    memo: &SqliteMemo,
    base: u64,
    tip: u64,
    id: u64,
    keys: &[u8],
    tombstones: &[Tombstone],
) -> Generation {
    let delta = delta_header(base, tip, id);
    let entries: Vec<_> = keys.iter().map(|&key| delta_entry(key, &delta)).collect();
    memo.write_delta(GenId(tip), GenId(id), &delta, &entries, tombstones)
        .unwrap()
}

#[test]
fn a_version_1_database_reads_and_takes_its_first_delta_in_place() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    memo.write_generation(None, GenId(1), &header(1), &encoded(3))
        .unwrap();
    assert_eq!(user_version(&raw(&db)), 1);
    assert!(!has_parent_column(&raw(&db)));

    let reopened = SqliteMemo::open(&db.path).unwrap();
    let chain = reopened.load_chain(&[]).unwrap().unwrap();
    assert_eq!(chain.depth(), 0);
    assert_eq!(chain.base().items().len(), 3);
    assert_eq!(reopened.load().unwrap().unwrap().id(), GenId(1));

    write_delta_on(&reopened, 1, 1, 2, &[7], &[]);
    let chain = reopened.load_chain(&[]).unwrap().unwrap();
    assert_eq!(chain.depth(), 1);
    assert_eq!(chain.base().items().len(), 3);
}

#[test]
fn the_version_becomes_2_on_the_first_delta_only() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    assert!(memo.supports_delta());
    memo.write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    memo.write_generation_streamed(
        Some(GenId(1)),
        GenId(2),
        &header(2),
        &mut encoded(2).into_iter().map(Ok),
    )
    .unwrap();
    assert_eq!(user_version(&raw(&db)), 1);
    assert!(!has_parent_column(&raw(&db)));

    write_delta_on(&memo, 2, 2, 3, &[9], &[]);
    assert_eq!(user_version(&raw(&db)), 2);
    assert!(has_parent_column(&raw(&db)));
    write_delta_on(&memo, 2, 3, 4, &[10], &[]);
    assert_eq!(user_version(&raw(&db)), 2);

    memo.write_generation(Some(GenId(4)), GenId(5), &header(5), &encoded(1))
        .unwrap();
    assert_eq!(user_version(&raw(&db)), 2);
    assert_eq!(memo.load_chain(&[]).unwrap().unwrap().depth(), 0);
}

#[test]
fn a_reader_that_wants_version_1_refuses_a_database_with_deltas() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    memo.write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    write_delta_on(&memo, 1, 1, 2, &[9], &[]);
    let mut conn = raw(&db);
    let Err(Error::Backend(err)) = create_schema_for(&mut conn, &[1]) else {
        panic!("a version 1 reader opened a database with deltas");
    };
    assert!(err.to_string().contains("user_version 2"), "got {err}");
    assert!(create_schema_for(&mut conn, &[1, 2]).is_ok());
}

#[test]
fn a_handle_opened_before_the_first_delta_reads_the_chain_another_handle_wrote() {
    let db = db();
    let early = SqliteMemo::open(&db.path).unwrap();
    early
        .write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    let late = SqliteMemo::open(&db.path).unwrap();
    write_delta_on(&late, 1, 1, 2, &[9], &[]);
    assert_eq!(early.load_chain(&[]).unwrap().unwrap().depth(), 1);
    write_delta_on(&early, 1, 2, 3, &[10], &[]);
    assert_eq!(late.load_chain(&[]).unwrap().unwrap().depth(), 2);
}

#[test]
fn tombstones_are_empty_frame_rows_and_load_as_tombstone_items() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    memo.write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    let tombstone = Tombstone {
        table: 0,
        key_hash: 0xabcd,
    };
    let written = write_delta_on(&memo, 1, 1, 2, &[5], &[tombstone]);
    let rows: i64 = raw(&db)
        .query_row(
            "SELECT count(*) FROM memo_entry WHERE generation = 2 AND len = 0 AND length(frame) = 0",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(rows, 1);
    let expected = IndexItem::tombstone(0, 0xabcd);
    assert!(written.items().contains(&expected));
    let loaded = memo.load_chain(&[]).unwrap().unwrap();
    assert!(loaded.tip().items().contains(&expected));
    assert_eq!(loaded.tip().items(), written.items());
}

#[test]
fn a_damaged_delta_loads_empty_and_a_known_layer_is_not_reloaded() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    let base = memo
        .write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    write_delta_on(&memo, 1, 1, 2, &[5], &[]);
    raw(&db)
        .execute("UPDATE memo_generation SET header = x'00' WHERE id = 2", [])
        .unwrap();
    let chain = memo.load_chain(&[]).unwrap().unwrap();
    assert_eq!(chain.depth(), 1);
    assert!(chain.tip().items().is_empty());
    assert_eq!(chain.tip().parent(), Some(GenId(1)));

    raw(&db)
        .execute("DELETE FROM memo_entry WHERE generation = 1", [])
        .unwrap();
    let known = [std::sync::Arc::new(base)];
    let chain = memo.load_chain(&known).unwrap().unwrap();
    assert_eq!(chain.base().items().len(), 2);
}

#[test]
fn a_streamed_full_write_from_the_tip_replaces_the_chain() {
    let db = db();
    let memo = SqliteMemo::open(&db.path).unwrap();
    memo.write_generation(None, GenId(1), &header(1), &encoded(2))
        .unwrap();
    write_delta_on(&memo, 1, 1, 2, &[5], &[]);
    write_delta_on(&memo, 1, 2, 3, &[6], &[]);
    memo.write_generation_streamed(
        Some(GenId(3)),
        GenId(4),
        &header(4),
        &mut encoded(1).into_iter().map(Ok),
    )
    .unwrap();
    assert_eq!(memo.load_chain(&[]).unwrap().unwrap().depth(), 0);
    let removing_old = memo.remove_generation(GenId(2));
    assert!(removing_old.is_ok());
    drop(memo.lock_checkpoint().unwrap());
    let ids: i64 = raw(&db)
        .query_row("SELECT count(*) FROM memo_generation", [], |row| row.get(0))
        .unwrap();
    assert_eq!(ids, 1);
}
