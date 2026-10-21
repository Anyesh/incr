use std::ffi::OsString;
use std::fs::{File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension, TransactionBehavior};

use super::{
    decode_header, encode_header, EncodedEntry, GenHeader, GenId, Generation, IndexItem,
    MemoBackend,
};
use crate::sqlite::{enable_wal, is_busy, sql};
use crate::{Error, Result, SqliteDurability};

/// Marks a database as a memo store, so that a log database or another
/// application's file is refused rather than written into.
const APPLICATION_ID: i32 = 0x696d_656d;
const SCHEMA_VERSION: i64 = 1;

// `memo_generation.header` is the generation id as 8 LE bytes followed by
// the `encode_header` frame, so a header moved to another row is detected.
// Ids, `key_hash` and `loc` are u64 stored by bit cast into SQLite's i64.
const SCHEMA: &str = "
    CREATE TABLE memo_generation (id INTEGER PRIMARY KEY, header BLOB NOT NULL);
    CREATE TABLE memo_current (
        one INTEGER PRIMARY KEY CHECK (one = 0),
        generation INTEGER NOT NULL
    );
    CREATE TABLE memo_entry (
        loc INTEGER PRIMARY KEY,
        generation INTEGER NOT NULL,
        tbl INTEGER NOT NULL,
        key_hash INTEGER NOT NULL,
        len INTEGER NOT NULL,
        frame BLOB NOT NULL
    );
    CREATE INDEX memo_entry_index ON memo_entry (generation, tbl, key_hash, len);
";

/// How [`SqliteMemo::open_with`] opens a memo store.
#[derive(Clone, Copy, Debug)]
pub struct SqliteMemoOptions {
    /// What a written generation survives. Under
    /// [`Buffered`](SqliteDurability::Buffered) a crash can lose the latest
    /// generations, and `load` then returns the newest whole one that
    /// survived.
    pub durability: SqliteDurability,
    /// How long a call waits for another connection's write lock before
    /// failing. A checkpoint that times out fails with
    /// [`Error::Busy`]. Defaults to five seconds.
    pub busy_timeout: Duration,
}

impl Default for SqliteMemoOptions {
    fn default() -> Self {
        SqliteMemoOptions {
            durability: SqliteDurability::Full,
            busy_timeout: Duration::from_secs(5),
        }
    }
}

/// A memo store in an SQLite database file of its own, shared by any number
/// of handles in any number of processes.
///
/// Each generation is a row of `memo_generation` and one row of
/// `memo_entry` per entry frame, whose rowid is the [`IndexItem::loc`];
/// `memo_current` names the current generation. A generation is written in
/// one transaction, which checks the base and sets `memo_current`, so a
/// crash leaves either the old generation current or the new one whole.
/// The database runs in WAL mode, and each handle keeps one connection for
/// reads and one for writes, so restores read the current generation while
/// a checkpoint writes the next one.
///
/// The checkpoint lock is an exclusive `flock` on a file next to the
/// database, named after it with `-checkpoint` appended; the OS releases it
/// when its holder's process dies. Like the `-wal` and `-shm` files, it
/// belongs to the database and is never removed. Taking it also removes
/// every generation but the current one.
#[derive(Debug)]
pub struct SqliteMemo {
    reader: Mutex<Connection>,
    writer: Mutex<Connection>,
    lock_path: PathBuf,
}

/// The checkpoint lock of a [`SqliteMemo`], released on drop.
#[derive(Debug)]
pub struct SqliteMemoGuard {
    _file: File,
}

impl SqliteMemo {
    /// Opens or creates the memo store at `path` with
    /// [`SqliteMemoOptions::default`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, SqliteMemoOptions::default())
    }

    /// Opens or creates the memo store at `path`, creating the schema if
    /// the file is new and switching it to WAL mode.
    ///
    /// Fails with [`Error::Backend`] when SQLite cannot open the file, or
    /// when it holds something other than a memo store of this schema
    /// version, such as a [`SqliteBackend`](crate::SqliteBackend) log.
    pub fn open_with(path: impl AsRef<Path>, options: SqliteMemoOptions) -> Result<Self> {
        let path = path.as_ref();
        // NO_MUTEX because each connection sits behind its own Mutex.
        let mut writer = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE,
        )
        .map_err(sql)?;
        writer.busy_timeout(options.busy_timeout).map_err(sql)?;
        enable_wal(&writer, options.busy_timeout)?;
        let synchronous = match options.durability {
            SqliteDurability::Full => "FULL",
            SqliteDurability::Buffered => "NORMAL",
        };
        writer
            .pragma_update(None, "synchronous", synchronous)
            .map_err(sql)?;
        create_schema(&mut writer)?;

        let reader = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_READ_WRITE,
        )
        .map_err(sql)?;
        reader.busy_timeout(options.busy_timeout).map_err(sql)?;

        let mut lock_path = OsString::from(path.as_os_str());
        lock_path.push("-checkpoint");
        Ok(SqliteMemo {
            reader: Mutex::new(reader),
            writer: Mutex::new(writer),
            lock_path: lock_path.into(),
        })
    }
}

// Every multi-statement change runs in a transaction that rolls back when a
// panic unwinds through it, so a poisoned connection is still consistent.
fn lock(conn: &Mutex<Connection>) -> MutexGuard<'_, Connection> {
    conn.lock().unwrap_or_else(PoisonError::into_inner)
}

impl MemoBackend for SqliteMemo {
    type Guard = SqliteMemoGuard;

    fn lock_checkpoint(&self) -> Result<SqliteMemoGuard> {
        // A file opened per call, because a flock does not exclude another
        // lock taken through the same open file.
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.lock_path)?;
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(Error::Busy),
            Err(TryLockError::Error(e)) => return Err(e.into()),
        }
        self.remove_leftovers()?;
        Ok(SqliteMemoGuard { _file: file })
    }

    fn load(&self) -> Result<Option<Generation>> {
        let mut conn = lock(&self.reader);
        // One read transaction, so a write or removal committing meanwhile
        // cannot pair one generation's header with another's index.
        let tx = conn.transaction().map_err(sql)?;
        let Some(id) = current(&tx)? else {
            return Ok(None);
        };
        let blob: Option<Vec<u8>> = tx
            .prepare_cached("SELECT header FROM memo_generation WHERE id = ?1")
            .map_err(sql)?
            .query_row([to_sql(id.0)], |row| row.get(0))
            .optional()
            .map_err(sql)?;
        let Some(header) = blob.as_deref().and_then(|blob| parse_header(id, blob)) else {
            return Ok(Some(Generation::empty(id)));
        };
        let mut select = tx
            .prepare_cached("SELECT loc, tbl, key_hash, len FROM memo_entry WHERE generation = ?1")
            .map_err(sql)?;
        let mut rows = select.query([to_sql(id.0)]).map_err(sql)?;
        let mut items = Vec::new();
        while let Some(row) = rows.next().map_err(sql)? {
            match parse_item(row) {
                Some(item) => items.push(item),
                None => return Ok(Some(Generation::empty(id))),
            }
        }
        Ok(Some(Generation::new(id, header, items)))
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        lock(&self.reader)
            .prepare_cached("SELECT frame FROM memo_entry WHERE loc = ?1 AND generation = ?2")
            .map_err(sql)?
            .query_row(params![to_sql(item.loc), to_sql(generation.0)], |row| {
                row.get(0)
            })
            .optional()
            .map_err(sql)
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        let mut blob = id.0.to_le_bytes().to_vec();
        blob.extend(encode_header(header)?);
        let mut conn = lock(&self.writer);
        let tx = match conn.transaction_with_behavior(TransactionBehavior::Immediate) {
            Err(e) if is_busy(&e) => return Err(Error::Busy),
            begun => begun.map_err(sql)?,
        };
        if current(&tx)? != base {
            return Err(Error::Busy);
        }
        let newest: Option<i64> = tx
            .query_row("SELECT max(id) FROM memo_generation", [], |row| row.get(0))
            .map_err(sql)?;
        if let Some(newest) = newest.map(from_sql).filter(|&newest| newest >= id.0) {
            return Err(Error::Backend(
                format!(
                    "memo generation {} is not newer than generation {newest}",
                    id.0
                )
                .into(),
            ));
        }
        tx.execute(
            "INSERT INTO memo_generation (id, header) VALUES (?1, ?2)",
            params![to_sql(id.0), blob],
        )
        .map_err(sql)?;
        let mut items = Vec::with_capacity(entries.len());
        {
            let mut insert = tx
                .prepare_cached(
                    "INSERT INTO memo_entry (generation, tbl, key_hash, len, frame) \
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(sql)?;
            for entry in entries {
                let len = u32::try_from(entry.frame.len())
                    .map_err(|_| Error::Corrupt("memo entry frame exceeds 4 GiB"))?;
                let loc = insert
                    .insert(params![
                        to_sql(id.0),
                        entry.table,
                        to_sql(entry.key_hash),
                        len,
                        entry.frame
                    ])
                    .map_err(sql)?;
                items.push(IndexItem {
                    table: entry.table,
                    key_hash: entry.key_hash,
                    loc: from_sql(loc),
                    len,
                });
            }
        }
        tx.execute(
            "INSERT OR REPLACE INTO memo_current (one, generation) VALUES (0, ?1)",
            [to_sql(id.0)],
        )
        .map_err(sql)?;
        tx.commit().map_err(sql)?;
        Ok(Generation::new(id, header.clone(), items))
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        let mut conn = lock(&self.writer);
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sql)?;
        if current(&tx)? == Some(generation) {
            return Err(Error::Backend(
                format!(
                    "memo generation {} is current and cannot be removed",
                    generation.0
                )
                .into(),
            ));
        }
        delete_generation(&tx, generation)?;
        tx.commit().map_err(sql)
    }
}

impl SqliteMemo {
    /// Removes every generation but the current one. Those below it are
    /// left by a removal that failed or never ran. Those above it only by a
    /// `memo_current` pointed back at an older generation, since a write
    /// commits its rows and the pointer together, and they would make every
    /// later write's id not newer than the newest row.
    fn remove_leftovers(&self) -> Result<()> {
        let mut conn = lock(&self.writer);
        if leftovers(&conn)?.is_empty() {
            return Ok(());
        }
        let tx = match conn.transaction_with_behavior(TransactionBehavior::Immediate) {
            Err(e) if is_busy(&e) => return Err(Error::Busy),
            begun => begun.map_err(sql)?,
        };
        for generation in leftovers(&tx)? {
            delete_generation(&tx, generation)?;
        }
        tx.commit().map_err(sql)
    }
}

fn leftovers(conn: &Connection) -> Result<Vec<GenId>> {
    let Some(current) = current(conn)? else {
        return Ok(Vec::new());
    };
    let mut select = conn
        .prepare_cached("SELECT id FROM memo_generation")
        .map_err(sql)?;
    let mut rows = select.query([]).map_err(sql)?;
    let mut ids = Vec::new();
    while let Some(row) = rows.next().map_err(sql)? {
        let id = GenId(from_sql(row.get(0).map_err(sql)?));
        if id != current {
            ids.push(id);
        }
    }
    Ok(ids)
}

fn delete_generation(conn: &Connection, generation: GenId) -> Result<()> {
    conn.execute(
        "DELETE FROM memo_entry WHERE generation = ?1",
        [to_sql(generation.0)],
    )
    .map_err(sql)?;
    conn.execute(
        "DELETE FROM memo_generation WHERE id = ?1",
        [to_sql(generation.0)],
    )
    .map_err(sql)?;
    Ok(())
}

fn create_schema(conn: &mut Connection) -> Result<()> {
    // Under the write lock, so that concurrent first opens create the
    // schema once and the losers see it.
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(sql)?;
    let application_id: i32 = tx
        .pragma_query_value(None, "application_id", |row| row.get(0))
        .map_err(sql)?;
    let version: i64 = tx
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(sql)?;
    match (application_id, version) {
        (0, 0) => {
            tx.execute_batch(SCHEMA).map_err(sql)?;
            tx.pragma_update(None, "application_id", APPLICATION_ID)
                .map_err(sql)?;
            tx.pragma_update(None, "user_version", SCHEMA_VERSION)
                .map_err(sql)?;
        }
        (APPLICATION_ID, SCHEMA_VERSION) => {}
        _ => {
            return Err(Error::Backend(
                format!(
                    "not an incr-persist memo store of schema version {SCHEMA_VERSION} \
                     (application_id {application_id:#x}, user_version {version})"
                )
                .into(),
            ))
        }
    }
    tx.commit().map_err(sql)
}

fn current(conn: &Connection) -> Result<Option<GenId>> {
    let id: Option<i64> = conn
        .prepare_cached("SELECT generation FROM memo_current WHERE one = 0")
        .map_err(sql)?
        .query_row([], |row| row.get(0))
        .optional()
        .map_err(sql)?;
    Ok(id.map(|id| GenId(from_sql(id))))
}

fn parse_header(id: GenId, blob: &[u8]) -> Option<GenHeader> {
    let (stored, frame) = blob.split_first_chunk::<8>()?;
    if u64::from_le_bytes(*stored) != id.0 {
        return None;
    }
    decode_header(frame).ok()
}

/// `None` when a column holds a value no write could have stored.
fn parse_item(row: &rusqlite::Row<'_>) -> Option<IndexItem> {
    let int = |column| row.get::<_, i64>(column).ok();
    Some(IndexItem {
        loc: from_sql(int(0)?),
        table: u32::try_from(int(1)?).ok()?,
        key_hash: from_sql(int(2)?),
        len: u32::try_from(int(3)?).ok()?,
    })
}

fn to_sql(value: u64) -> i64 {
    value as i64
}

fn from_sql(value: i64) -> u64 {
    value as u64
}

#[cfg(test)]
#[path = "sqlite_tests.rs"]
mod tests;
