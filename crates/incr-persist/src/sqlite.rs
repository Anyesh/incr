use std::collections::hash_map::RandomState;
use std::ffi::OsString;
use std::fs::{File, OpenOptions, TryLockError};
use std::hash::BuildHasher;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use rusqlite::{
    params, Connection, ErrorCode, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};

use crate::{Backend, Error, Position, ReadOutcome, Record, Result};

const SCHEMA_VERSION: i64 = 1;

/// How much a committed append survives, chosen when a writable
/// [`SqliteBackend`] is opened.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SqliteDurability {
    /// `synchronous=FULL`: `append` returns only after its transaction is
    /// fsynced, so it survives power loss and `sync` has nothing to do.
    #[default]
    Full,
    /// `synchronous=NORMAL`: an append survives a crash of the process, but
    /// the latest ones can roll back after power loss or an OS crash until
    /// SQLite next checkpoints its write-ahead log. `sync` does not close
    /// that window. Commits skip the fsync, which makes them much cheaper.
    Buffered,
}

/// How [`SqliteBackend::open_with`] opens a log.
#[derive(Clone, Copy, Debug)]
pub struct SqliteOptions {
    /// Durability of appends. Ignored on a read-only open.
    pub durability: SqliteDurability,
    /// How long a call waits for another connection's lock (a writer's, or
    /// a reader's during write-ahead log recovery) before failing. Defaults
    /// to five seconds. Writers of this crate take turns, so under
    /// continuous contention a write waits for the turns of the writers
    /// ahead of it, not for one of them to stop.
    pub busy_timeout: Duration,
    /// Open without write access: the database must already exist, and
    /// `append` and `compact` fail with [`Error::ReadOnly`].
    pub read_only: bool,
}

impl Default for SqliteOptions {
    fn default() -> Self {
        SqliteOptions {
            durability: SqliteDurability::Full,
            busy_timeout: Duration::from_secs(5),
            read_only: false,
        }
    }
}

/// A log in an SQLite database file, shared by any number of writers in
/// any number of processes.
///
/// Records live one per row in `log(pos INTEGER PRIMARY KEY, rec BLOB)`,
/// each as [`Record::encode`] bytes, and the snapshot lives in the single
/// row of `snapshot(pos, recs)`. An append numbers its records from just
/// above the last row, or above the snapshot's position when compaction
/// left no rows, so no position is handed out twice. The database runs in
/// WAL mode, so readers never wait for writers. `append` and `compact` take
/// SQLite's write lock with `BEGIN IMMEDIATE` and hold it until commit, so
/// positions increase in commit order. A read sees a prefix of the commits,
/// which is therefore a prefix of the positions, and that is what makes
/// `read_from` gap-free across processes.
///
/// Each handle owns one connection and runs its calls on it one at a time.
/// Threads that should append in parallel open a handle each. A call that
/// finds the database locked by another connection retries for
/// [`SqliteOptions::busy_timeout`] before failing. `sync` is a no-op; see
/// [`SqliteDurability`] for what an append survives.
///
/// Writable handles take turns at the write lock, so one that writes
/// continuously cannot shut the others out: a writer that has to wait
/// registers in a file next to the database, named after it with
/// `-waiters` appended, and a writer that has held the lock for a while
/// stands aside while anyone is registered. Like the `-wal` and `-shm`
/// files, it belongs to the database and is never removed.
#[derive(Debug)]
pub struct SqliteBackend {
    inner: Mutex<Inner>,
    writable: bool,
}

#[derive(Debug)]
struct Inner {
    conn: Connection,
    /// `None` on a read-only handle.
    turns: Option<Turns>,
}

// The snapshot row's `recs` frames each encoded record as a u64 LE length
// followed by its bytes. `user_version` versions this layout and the
// tables together.
const SCHEMA: &str = "
    CREATE TABLE log (pos INTEGER PRIMARY KEY, rec BLOB NOT NULL);
    CREATE TABLE snapshot (
        id INTEGER PRIMARY KEY CHECK (id = 0),
        pos INTEGER NOT NULL,
        recs BLOB NOT NULL
    );
";

impl SqliteBackend {
    /// Opens or creates the log at `path` with [`SqliteOptions::default`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, SqliteOptions::default())
    }

    /// Opens an existing log at `path` read-only.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(
            path,
            SqliteOptions {
                read_only: true,
                ..SqliteOptions::default()
            },
        )
    }

    /// Opens the log at `path` as `options` describe. A writable open
    /// creates the file and the schema if they are missing and switches the
    /// database to WAL mode.
    ///
    /// Fails with [`Error::Backend`] when SQLite cannot open the file, when
    /// the file holds a schema version this build does not read, or when a
    /// read-only open finds no log there.
    pub fn open_with(path: impl AsRef<Path>, options: SqliteOptions) -> Result<Self> {
        // NO_MUTEX because the handle's own Mutex already serializes every
        // use of the connection.
        let flags = OpenFlags::SQLITE_OPEN_NO_MUTEX
            | if options.read_only {
                OpenFlags::SQLITE_OPEN_READ_ONLY
            } else {
                OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE
            };
        let path = path.as_ref();
        let conn = Connection::open_with_flags(path, flags).map_err(sql)?;
        conn.busy_timeout(options.busy_timeout).map_err(sql)?;
        let turns = if options.read_only {
            match schema_version(&conn)? {
                SCHEMA_VERSION => {}
                other => return Err(unsupported_schema(other)),
            }
            None
        } else {
            enable_wal(&conn, options.busy_timeout)?;
            let synchronous = match options.durability {
                SqliteDurability::Full => "FULL",
                SqliteDurability::Buffered => "NORMAL",
            };
            conn.pragma_update(None, "synchronous", synchronous)
                .map_err(sql)?;
            let mut turns = Turns::open(path, options.busy_timeout)?;
            create_schema(&conn, &mut turns)?;
            Some(turns)
        };
        Ok(SqliteBackend {
            inner: Mutex::new(Inner { conn, turns }),
            writable: !options.read_only,
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // Recovering from poison is sound because every multi-statement
        // change runs in a transaction whose guard rolls it back when a
        // panic unwinds through it.
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn require_writable(&self) -> Result<()> {
        if self.writable {
            Ok(())
        } else {
            Err(Error::ReadOnly)
        }
    }
}

impl Backend for SqliteBackend {
    fn append(&self, records: &[Record]) -> Result<Position> {
        self.require_writable()?;
        if records.is_empty() {
            return tail(&self.lock().conn);
        }
        let encoded = records
            .iter()
            .map(Record::encode)
            .collect::<Result<Vec<_>>>()?;
        let mut inner = self.lock();
        let tx = inner.begin_write()?;
        let mut last = sql_position(tail(&tx)?);
        {
            let mut insert = tx
                .prepare_cached("INSERT INTO log (pos, rec) VALUES (?1, ?2)")
                .map_err(sql)?;
            for rec in &encoded {
                last += 1;
                insert.execute(params![last, rec]).map_err(sql)?;
            }
        }
        tx.commit().map_err(sql)?;
        position(last)
    }

    fn sync(&self, _upto: Position) -> Result<()> {
        Ok(())
    }

    fn read_from(&self, after: Position, out: &mut Vec<(Position, Record)>) -> Result<ReadOutcome> {
        let mut inner = self.lock();
        // One read transaction for both queries, so that a compaction
        // committing between them cannot pair the old snapshot position
        // with the trimmed log.
        let tx = inner.conn.transaction().map_err(sql)?;
        if let Some(snapshot_at) = snapshot_position(&tx)? {
            if after < snapshot_at {
                return Ok(ReadOutcome::Compacted { snapshot_at });
            }
        }
        let start = out.len();
        let pushed = push_records_after(&tx, after, out);
        if pushed.is_err() {
            out.truncate(start);
        }
        pushed.map(|()| ReadOutcome::Records)
    }

    fn load_snapshot(&self) -> Result<Option<(Position, Vec<Record>)>> {
        let row: Option<(i64, Vec<u8>)> = self
            .lock()
            .conn
            .query_row("SELECT pos, recs FROM snapshot", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .optional()
            .map_err(sql)?;
        row.map(|(at, recs)| Ok((position(at)?, decode_snapshot(&recs)?)))
            .transpose()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> Result<()> {
        self.require_writable()?;
        let recs = encode_snapshot(&folded)?;
        let mut inner = self.lock();
        let tx = inner.begin_write()?;
        let snapshot_at = snapshot_position(&tx)?.unwrap_or(Position::ZERO);
        let tail = tail(&tx)?;
        if upto < snapshot_at || upto > tail {
            return Err(Error::InvalidCompaction {
                upto,
                snapshot_at,
                tail,
            });
        }
        let upto_sql = sql_position(upto);
        tx.execute("DELETE FROM log WHERE pos <= ?1", [upto_sql])
            .map_err(sql)?;
        tx.execute(
            "INSERT OR REPLACE INTO snapshot (id, pos, recs) VALUES (0, ?1, ?2)",
            params![upto_sql, recs],
        )
        .map_err(sql)?;
        tx.commit().map_err(sql)
    }

    fn writable(&self) -> bool {
        self.writable
    }
}

impl Inner {
    fn begin_write(&mut self) -> Result<Transaction<'_>> {
        let turns = self.turns.as_mut().ok_or(Error::ReadOnly)?;
        turns.begin(&self.conn)
    }
}

/// How long a writer may keep beginning transactions back to back before it
/// checks for waiting writers. The check is two `flock` calls, so on an
/// uncontended log it costs next to nothing at this interval, and a writer
/// that commits without fsync gets many commits per turn, which amortizes
/// the handoff.
const TURN: Duration = Duration::from_millis(2);

/// Bounds of the random pause between attempts at the write lock. It does
/// not grow with the wait, because growing backoff favors whoever started
/// waiting last, and it is random so that no waiter's retries settle into a
/// phase that always lands just after the holder's commits.
const PAUSE_MIN: Duration = Duration::from_micros(50);
const PAUSE_MAX: Duration = Duration::from_micros(250);

/// The longest a writer stands aside for registered waiters that never take
/// the lock, such as one in a stopped process.
const STAND_ASIDE_LIMIT: Duration = Duration::from_millis(20);

/// One writable handle's part in the turn taking among a log's writers.
///
/// SQLite's busy handler retries a locked `BEGIN IMMEDIATE` at intervals
/// that grow to 100 ms, and a writer committing back to back frees the lock
/// for only microseconds between commits that take milliseconds under
/// `synchronous=FULL`, so a waiter left to it can sleep through every
/// opening until its busy timeout fails it. Instead, a waiter holds a shared
/// lock on the `-waiters` file while it retries every
/// `PAUSE_MIN..PAUSE_MAX`, and a writer whose turn has run out finds that
/// lock and waits for another writer to commit before beginning again. The
/// OS drops a waiter's lock when its process dies.
#[derive(Debug)]
struct Turns {
    waiters: File,
    busy_timeout: Duration,
    turn_started: Instant,
    /// `data_version` when this writer last stood aside and no waiter took
    /// the lock. Until another writer commits, the registered waiters are
    /// presumed unable to and are not waited for again.
    ignored_waiters: Option<i64>,
    rng: u64,
}

impl Turns {
    fn open(db: &Path, busy_timeout: Duration) -> Result<Self> {
        let mut path = OsString::from(db.as_os_str());
        path.push("-waiters");
        let waiters = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        Ok(Turns {
            waiters,
            busy_timeout,
            turn_started: Instant::now(),
            ignored_waiters: None,
            // xorshift needs a nonzero state.
            rng: RandomState::new().hash_one(0u8) | 1,
        })
    }

    /// Begins an immediate transaction on `conn`. Fails with SQLite's busy
    /// error when the write lock stays out of reach for `busy_timeout`.
    fn begin<'c>(&mut self, conn: &'c Connection) -> Result<Transaction<'c>> {
        let deadline = Instant::now() + self.busy_timeout;
        if self.turn_started.elapsed() >= TURN {
            self.stand_aside(conn, deadline)?;
            self.turn_started = Instant::now();
        }
        // Off during the attempts, because its sleeps would miss the
        // openings that `acquire` retries for.
        conn.busy_timeout(Duration::ZERO).map_err(sql)?;
        let begun = self.acquire(conn, deadline);
        conn.busy_timeout(self.busy_timeout).map_err(sql)?;
        begun
    }

    fn acquire<'c>(&mut self, conn: &'c Connection, deadline: Instant) -> Result<Transaction<'c>> {
        let mut registered = false;
        let begun = loop {
            match Transaction::new_unchecked(conn, TransactionBehavior::Immediate) {
                Err(e) if is_busy(&e) && Instant::now() < deadline => {
                    if !registered {
                        if let Err(e) = self.waiters.lock_shared() {
                            break Err(e.into());
                        }
                        registered = true;
                    }
                    let left = deadline.saturating_duration_since(Instant::now());
                    thread::sleep(self.pause().min(left));
                }
                other => break other.map_err(sql),
            }
        };
        if registered {
            self.waiters.unlock()?;
            // The holder that made this writer wait has had its turn.
            self.turn_started = Instant::now();
        }
        begun
    }

    /// When writers are registered as waiting, waits until one of them
    /// commits or none is left, so that this writer does not take the lock
    /// straight back.
    fn stand_aside(&mut self, conn: &Connection, deadline: Instant) -> Result<()> {
        if !self.someone_waiting()? {
            return Ok(());
        }
        let version = data_version(conn)?;
        if self.ignored_waiters == Some(version) {
            return Ok(());
        }
        let until = deadline.min(Instant::now() + STAND_ASIDE_LIMIT);
        while Instant::now() < until {
            thread::sleep(self.pause());
            if !self.someone_waiting()? || data_version(conn)? != version {
                return Ok(());
            }
        }
        self.ignored_waiters = Some(version);
        Ok(())
    }

    fn someone_waiting(&self) -> Result<bool> {
        match self.waiters.try_lock() {
            Ok(()) => {
                self.waiters.unlock()?;
                Ok(false)
            }
            Err(TryLockError::WouldBlock) => Ok(true),
            Err(TryLockError::Error(e)) => Err(e.into()),
        }
    }

    fn pause(&mut self) -> Duration {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let span = (PAUSE_MAX - PAUSE_MIN).as_nanos() as u64;
        PAUSE_MIN + Duration::from_nanos(self.rng % span)
    }
}

fn is_busy(error: &rusqlite::Error) -> bool {
    error.sqlite_error_code() == Some(ErrorCode::DatabaseBusy)
}

/// Changes whenever another connection has committed since the last call
/// on `conn`.
fn data_version(conn: &Connection) -> Result<i64> {
    conn.pragma_query_value(None, "data_version", |row| row.get(0))
        .map_err(sql)
}

fn enable_wal(conn: &Connection, busy_timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + busy_timeout;
    let mode: String = loop {
        match conn.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0)) {
            Ok(mode) => break mode,
            // Leaving rollback mode needs an exclusive lock, and when
            // concurrent first opens contend for it SQLite reports
            // SQLITE_BUSY at once instead of calling the busy handler, so
            // the wait busy_timeout promises has to happen here.
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == ErrorCode::DatabaseBusy && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(1));
            }
            Err(e) => return Err(sql(e)),
        }
    };
    if mode.eq_ignore_ascii_case("wal") {
        Ok(())
    } else {
        Err(Error::Backend(
            format!("SQLite refused WAL mode and stayed in {mode} mode").into(),
        ))
    }
}

fn create_schema(conn: &Connection, turns: &mut Turns) -> Result<()> {
    // Under the write lock, so that concurrent first opens create the
    // schema once and the losers see it.
    let tx = turns.begin(conn)?;
    match schema_version(&tx)? {
        0 => {
            tx.execute_batch(SCHEMA).map_err(sql)?;
            tx.pragma_update(None, "user_version", SCHEMA_VERSION)
                .map_err(sql)?;
        }
        SCHEMA_VERSION => {}
        other => return Err(unsupported_schema(other)),
    }
    tx.commit().map_err(sql)
}

fn schema_version(conn: &Connection) -> Result<i64> {
    conn.pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(sql)
}

fn unsupported_schema(found: i64) -> Error {
    Error::Backend(
        format!(
            "unsupported incr-persist SQLite schema version {found} (this build reads \
             {SCHEMA_VERSION}; 0 means no writable open ever created the log)"
        )
        .into(),
    )
}

/// The highest position ever handed out, compacted ones included: the last
/// row, or the snapshot's position when there are no rows, because
/// compaction drops only rows at or below the position it records.
///
/// Computed rather than kept by AUTOINCREMENT, whose `sqlite_sequence`
/// update would add a page to the write-ahead log on every commit.
fn tail(conn: &Connection) -> Result<Position> {
    let at = conn
        .prepare_cached(
            "SELECT max(ifnull((SELECT max(pos) FROM log), 0), \
                        ifnull((SELECT pos FROM snapshot), 0))",
        )
        .map_err(sql)?
        .query_row([], |row| row.get(0))
        .map_err(sql)?;
    position(at)
}

fn snapshot_position(conn: &Connection) -> Result<Option<Position>> {
    let at: Option<i64> = conn
        .query_row("SELECT pos FROM snapshot", [], |row| row.get(0))
        .optional()
        .map_err(sql)?;
    at.map(position).transpose()
}

fn push_records_after(
    conn: &Connection,
    after: Position,
    out: &mut Vec<(Position, Record)>,
) -> Result<()> {
    let mut select = conn
        .prepare_cached("SELECT pos, rec FROM log WHERE pos > ?1 ORDER BY pos")
        .map_err(sql)?;
    let mut rows = select.query([sql_position(after)]).map_err(sql)?;
    while let Some(row) = rows.next().map_err(sql)? {
        let at = position(row.get(0).map_err(sql)?)?;
        let rec = row
            .get_ref(1)
            .map_err(sql)?
            .as_blob()
            .map_err(|e| Error::Backend(Box::new(e)))?;
        out.push((at, Record::decode(rec)?));
    }
    Ok(())
}

fn encode_snapshot(records: &[Record]) -> Result<Vec<u8>> {
    let mut recs = Vec::new();
    for record in records {
        let encoded = record.encode()?;
        recs.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        recs.extend_from_slice(&encoded);
    }
    Ok(recs)
}

fn decode_snapshot(mut recs: &[u8]) -> Result<Vec<Record>> {
    let mut records = Vec::new();
    while !recs.is_empty() {
        let (len, rest) = recs
            .split_first_chunk::<8>()
            .ok_or(Error::Corrupt("truncated snapshot frame length"))?;
        let len = usize::try_from(u64::from_le_bytes(*len))
            .map_err(|_| Error::Corrupt("snapshot frame length overflows usize"))?;
        if rest.len() < len {
            return Err(Error::Corrupt("truncated snapshot frame"));
        }
        let (encoded, rest) = rest.split_at(len);
        records.push(Record::decode(encoded)?);
        recs = rest;
    }
    Ok(records)
}

fn position(raw: i64) -> Result<Position> {
    u64::try_from(raw)
        .map(Position)
        .map_err(|_| Error::Backend(format!("negative log position {raw}").into()))
}

/// Converts for comparison against stored positions, which never exceed
/// `i64::MAX`, so clamping larger ones keeps every comparison's outcome.
fn sql_position(at: Position) -> i64 {
    i64::try_from(at.0).unwrap_or(i64::MAX)
}

fn sql(error: rusqlite::Error) -> Error {
    Error::Backend(Box::new(error))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Barrier;
    use std::thread;

    use super::*;

    fn set(value: u8) -> Record {
        Record::Set {
            key: "a".into(),
            value: vec![value],
        }
    }

    fn pragma_i64(backend: &SqliteBackend, name: &str) -> i64 {
        backend
            .lock()
            .conn
            .pragma_query_value(None, name, |row| row.get(0))
            .unwrap()
    }

    #[test]
    fn durability_and_busy_timeout_reach_the_connection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.sqlite");
        for (durability, synchronous) in
            [(SqliteDurability::Full, 2), (SqliteDurability::Buffered, 1)]
        {
            let options = SqliteOptions {
                durability,
                busy_timeout: Duration::from_millis(1234),
                ..SqliteOptions::default()
            };
            let backend = SqliteBackend::open_with(&path, options).unwrap();
            assert_eq!(pragma_i64(&backend, "synchronous"), synchronous);
            assert_eq!(pragma_i64(&backend, "busy_timeout"), 1234);
            let mode: String = backend
                .lock()
                .conn
                .pragma_query_value(None, "journal_mode", |row| row.get(0))
                .unwrap();
            assert_eq!(mode, "wal");
        }
    }

    #[test]
    fn concurrent_first_opens_share_one_log() {
        const OPENERS: usize = 8;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.sqlite");
        let barrier = Barrier::new(OPENERS);
        let returned: Vec<Position> = thread::scope(|s| {
            let joins: Vec<_> = (0..OPENERS)
                .map(|i| {
                    let (path, barrier) = (&path, &barrier);
                    s.spawn(move || {
                        barrier.wait();
                        let backend = SqliteBackend::open(path).unwrap();
                        backend.append(&[set(i as u8)]).unwrap()
                    })
                })
                .collect();
            joins.into_iter().map(|j| j.join().unwrap()).collect()
        });
        let mut sorted = returned.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            (1..=OPENERS as u64).map(Position).collect::<Vec<_>>()
        );

        let mut out = Vec::new();
        let backend = SqliteBackend::open(&path).unwrap();
        backend.read_from(Position::ZERO, &mut out).unwrap();
        assert_eq!(out.len(), OPENERS);
    }

    /// Polls `reader` until `finished` is set and a read then finds nothing
    /// new, asserting that every run starts right after what it applied.
    /// Positions in this backend are dense, so any other start is a skip.
    fn follow_without_gaps(reader: &SqliteBackend, finished: &AtomicBool) -> Position {
        let mut applied = Position::ZERO;
        let mut out = Vec::new();
        loop {
            let done = finished.load(Ordering::Acquire);
            out.clear();
            match reader.read_from(applied, &mut out).unwrap() {
                ReadOutcome::Records => {
                    for (offset, (position, _)) in out.iter().enumerate() {
                        assert_eq!(
                            position.0,
                            applied.0 + 1 + offset as u64,
                            "read_from({applied}) skipped a position"
                        );
                    }
                    match out.last() {
                        Some((last, _)) => applied = *last,
                        None if done => return applied,
                        None => {}
                    }
                }
                ReadOutcome::Compacted { snapshot_at } => {
                    assert!(snapshot_at > applied);
                    let (loaded, _) = reader.load_snapshot().unwrap().unwrap();
                    assert!(loaded >= snapshot_at);
                    applied = loaded;
                }
            }
        }
    }

    /// A compaction committing while a reader is between its snapshot check
    /// and its log query must reach that reader as `Compacted`, never as a
    /// run that starts above the reader's position.
    #[test]
    fn readers_racing_compaction_see_no_gap() {
        const APPENDS: u32 = 2000;
        const READERS: usize = 4;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.sqlite");
        let buffered = SqliteOptions {
            durability: SqliteDurability::Buffered,
            ..SqliteOptions::default()
        };
        let appender = SqliteBackend::open_with(&path, buffered).unwrap();
        let compactor = SqliteBackend::open_with(&path, buffered).unwrap();
        let readers: Vec<_> = (0..READERS)
            .map(|_| SqliteBackend::open_read_only(&path).unwrap())
            .collect();
        let appended = AtomicBool::new(false);
        let finished = AtomicBool::new(false);

        let (compactions, followed) = thread::scope(|s| {
            s.spawn(|| {
                for value in 0..APPENDS {
                    let value = value as u8;
                    appender.append(&[set(value), set(value)]).unwrap();
                }
                appended.store(true, Ordering::Release);
            });
            let compacting = s.spawn(|| {
                let mut compactions = 0;
                while !appended.load(Ordering::Acquire) {
                    // One below the tail, so the log keeps a record above
                    // the new snapshot for a torn read to land on.
                    let tail = compactor.append(&[]).unwrap();
                    let upto = Position(tail.0.saturating_sub(1));
                    compactor.compact(upto, vec![]).unwrap();
                    compactions += 1;
                }
                finished.store(true, Ordering::Release);
                compactions
            });
            let following: Vec<_> = readers
                .iter()
                .map(|reader| s.spawn(|| follow_without_gaps(reader, &finished)))
                .collect();
            let followed: Vec<Position> =
                following.into_iter().map(|j| j.join().unwrap()).collect();
            (compacting.join().unwrap(), followed)
        });
        assert!(compactions > 0);
        assert_eq!(followed, vec![Position(2 * APPENDS as u64); READERS]);
    }

    /// A registration whose owner never takes the lock, as when its
    /// process is stopped, costs a writer one stand-aside rather than one
    /// per turn.
    #[test]
    fn a_waiter_that_never_takes_the_lock_is_stood_aside_for_once() {
        const APPENDS: u32 = 50;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.sqlite");
        let backend = SqliteBackend::open_with(
            &path,
            SqliteOptions {
                durability: SqliteDurability::Buffered,
                ..SqliteOptions::default()
            },
        )
        .unwrap();
        let stalled = File::options()
            .write(true)
            .open(dir.path().join("log.sqlite-waiters"))
            .unwrap();
        stalled.lock_shared().unwrap();

        let start = Instant::now();
        for value in 0..APPENDS {
            thread::sleep(TURN);
            backend.append(&[set(value as u8)]).unwrap();
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < STAND_ASIDE_LIMIT * APPENDS / 2,
            "{APPENDS} appends a turn apart took {elapsed:?}"
        );
    }

    #[test]
    fn read_only_open_follows_a_log_after_every_writer_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.sqlite");
        let tail = {
            let writer = SqliteBackend::open(&path).unwrap();
            writer.append(&[set(1), set(2)]).unwrap()
        };
        let reader = SqliteBackend::open_read_only(&path).unwrap();
        assert!(!reader.writable());
        let mut out = Vec::new();
        reader.read_from(Position::ZERO, &mut out).unwrap();
        assert_eq!(out.last().map(|(at, _)| *at), Some(tail));
    }

    #[test]
    fn read_only_open_of_a_missing_log_fails_without_creating_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.sqlite");
        assert!(SqliteBackend::open_read_only(&path).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn unknown_schema_version_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.sqlite");
        drop(SqliteBackend::open(&path).unwrap());
        Connection::open(&path)
            .unwrap()
            .pragma_update(None, "user_version", SCHEMA_VERSION + 1)
            .unwrap();
        assert!(matches!(SqliteBackend::open(&path), Err(Error::Backend(_))));
        assert!(matches!(
            SqliteBackend::open_read_only(&path),
            Err(Error::Backend(_))
        ));
    }

    #[test]
    fn undecodable_row_fails_the_read_and_leaves_out_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.sqlite");
        let backend = SqliteBackend::open(&path).unwrap();
        backend.append(&[set(1)]).unwrap();
        Connection::open(&path)
            .unwrap()
            .execute("INSERT INTO log (rec) VALUES (x'ff00')", [])
            .unwrap();

        let mut out = vec![(Position(u64::MAX), set(9))];
        assert!(matches!(
            backend.read_from(Position::ZERO, &mut out),
            Err(Error::UnsupportedFormat { found: 0xff, .. })
        ));
        assert_eq!(out, vec![(Position(u64::MAX), set(9))]);
    }
}
