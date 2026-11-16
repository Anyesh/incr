use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use rusqlite::{
    params, params_from_iter, Connection, OpenFlags, OptionalExtension, TransactionBehavior,
};

use super::{
    decode_delta_header, decode_header, encode_delta_header, encode_header, Chain, DeltaHeader,
    EncodedEntry, GenHeader, GenId, Generation, IndexItem, MemoBackend, Tombstone,
};
use crate::flock::Flock;
use crate::sqlite::{enable_wal, is_busy, sql};
use crate::{Error, Result, SqliteDurability};

/// Marks a database as a memo store, so that a log database or another
/// application's file is refused rather than written into.
const APPLICATION_ID: i32 = 0x696d_656d;
const SCHEMA_VERSION: i64 = 1;
/// Set by the first delta, which also adds `memo_generation.parent`. A reader
/// that knows only [`SCHEMA_VERSION`] refuses the database when it opens.
const DELTA_SCHEMA_VERSION: i64 = 2;
const READ_BATCH: usize = 256;

// `memo_generation.header` is the generation id as 8 LE bytes followed by
// the `encode_header` frame (`encode_delta_header` for a delta), so a header
// moved to another row is detected.
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
/// every generation outside the current chain.
///
/// A delta is a row of `memo_generation` with its parent's id in `parent`,
/// whose entry rows hold the delta's frames and its tombstones, rows with an
/// empty frame. The first delta adds the `parent` column and sets
/// `user_version` to 2, in the transaction that writes it; a database that
/// never held one stays at version 1 and reads as before. Each read checks
/// `user_version` inside its own transaction, so a handle sees a chain
/// another handle began after it opened. A reader built before deltas checks
/// the version only when it opens: one already open keeps reading, and
/// since it cannot parse a delta's header it loads an empty generation and
/// misses, never a wrong value.
#[derive(Debug)]
pub struct SqliteMemo {
    reader: Mutex<Connection>,
    writer: Mutex<Connection>,
    lock_path: PathBuf,
}

/// The checkpoint lock of a [`SqliteMemo`], released on drop.
#[derive(Debug)]
pub struct SqliteMemoGuard {
    _file: Flock,
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
        // After the schema check, because the journal mode is stored in the
        // file and a refused file must be left as it was.
        create_schema(&mut writer)?;
        enable_wal(&writer, options.busy_timeout)?;
        let synchronous = match options.durability {
            SqliteDurability::Full => "FULL",
            SqliteDurability::Buffered => "NORMAL",
        };
        writer
            .pragma_update(None, "synchronous", synchronous)
            .map_err(sql)?;

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
        Ok(SqliteMemoGuard { _file: Flock(file) })
    }

    fn load(&self) -> Result<Option<Generation>> {
        let mut conn = lock(&self.reader);
        // One read transaction, so a write or removal committing meanwhile
        // cannot pair one generation's header with another's index.
        let tx = conn.transaction().map_err(sql)?;
        let Some(rows) = chain_rows(&tx)? else {
            return Ok(None);
        };
        Ok(Some(layer(&tx, &rows[0], rows[0].id, None)?))
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        debug_assert!(!item.is_tombstone(), "reading a tombstone's frame");
        lock(&self.reader)
            .prepare_cached("SELECT frame FROM memo_entry WHERE loc = ?1 AND generation = ?2")
            .map_err(sql)?
            .query_row(params![to_sql(item.loc), to_sql(generation.0)], |row| {
                row.get(0)
            })
            .optional()
            .map_err(sql)
    }

    fn read_entries(&self, generation: GenId, items: &[IndexItem]) -> Result<Vec<Option<Vec<u8>>>> {
        debug_assert!(
            items.iter().all(|item| !item.is_tombstone()),
            "reading a tombstone's frame"
        );
        let mut conn = lock(&self.reader);
        let tx = conn.transaction().map_err(sql)?;
        let mut frames: HashMap<u64, Vec<u8>> = HashMap::with_capacity(items.len());
        for chunk in items.chunks(READ_BATCH) {
            let mut select = tx
                .prepare_cached(&read_entries_sql(chunk.len()))
                .map_err(sql)?;
            let args = std::iter::once(to_sql(generation.0))
                .chain(chunk.iter().map(|item| to_sql(item.loc)));
            let mut rows = select.query(params_from_iter(args)).map_err(sql)?;
            while let Some(row) = rows.next().map_err(sql)? {
                frames.insert(from_sql(row.get(0).map_err(sql)?), row.get(1).map_err(sql)?);
            }
        }
        Ok(items
            .iter()
            .map(|item| frames.get(&item.loc).cloned())
            .collect())
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        self.write_generation_streamed(base, id, header, &mut entries.iter().cloned().map(Ok))
    }

    /// Pulls inside the write transaction, which only the writer connection
    /// holds, so the iterator's reads through this handle take the reader
    /// connection and do not wait.
    fn write_generation_streamed(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &mut dyn Iterator<Item = Result<EncodedEntry>>,
    ) -> Result<Generation> {
        let mut blob = id.0.to_le_bytes().to_vec();
        blob.extend(encode_header(header)?);
        let items = self.write_layer(base, id, &blob, None, entries, &[])?;
        Ok(Generation::new(id, header.clone(), items))
    }

    fn supports_delta(&self) -> bool {
        true
    }

    fn load_chain(&self, known: &[Arc<Generation>]) -> Result<Option<Chain>> {
        let mut conn = lock(&self.reader);
        let tx = conn.transaction().map_err(sql)?;
        let Some(rows) = chain_rows(&tx)? else {
            return Ok(None);
        };
        let base = rows[0].id;
        let layers = rows
            .iter()
            .map(
                |row| match known.iter().find(|layer| layer.id() == row.id) {
                    Some(layer) => Ok(Arc::clone(layer)),
                    None => layer(&tx, row, base, row.parent).map(Arc::new),
                },
            )
            .collect::<Result<Vec<_>>>()?;
        Chain::new(layers).map(Some)
    }

    fn write_delta(
        &self,
        tip: GenId,
        id: GenId,
        header: &DeltaHeader,
        entries: &[EncodedEntry],
        tombstones: &[Tombstone],
    ) -> Result<Generation> {
        if header.parent != tip {
            return Err(Error::Backend(
                "a memo delta's parent is not the tip it is written on".into(),
            ));
        }
        let mut blob = id.0.to_le_bytes().to_vec();
        blob.extend(encode_delta_header(header)?);
        let items = self.write_layer(
            Some(tip),
            id,
            &blob,
            Some(tip),
            &mut entries.iter().cloned().map(Ok),
            tombstones,
        )?;
        Ok(Generation::new_delta(id, header.clone(), items))
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        remove_in_one(&mut lock(&self.writer), &[generation])
    }

    fn remove_generations(&self, generations: &[GenId]) -> Vec<GenId> {
        match remove_in_one(&mut lock(&self.writer), generations) {
            Ok(()) => Vec::new(),
            Err(_) => generations.to_vec(),
        }
    }
}

impl SqliteMemo {
    /// Writes one layer, a full generation or a delta, in one transaction
    /// that checks `tip` first, and returns its index items.
    fn write_layer(
        &self,
        tip: Option<GenId>,
        id: GenId,
        header: &[u8],
        parent: Option<GenId>,
        entries: &mut dyn Iterator<Item = Result<EncodedEntry>>,
        tombstones: &[Tombstone],
    ) -> Result<Vec<IndexItem>> {
        let mut conn = lock(&self.writer);
        let tx = match conn.transaction_with_behavior(TransactionBehavior::Immediate) {
            Err(e) if is_busy(&e) => return Err(Error::Busy),
            begun => begun.map_err(sql)?,
        };
        if current(&tx)? != tip {
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
        if parent.is_some() && user_version(&tx)? < DELTA_SCHEMA_VERSION {
            tx.execute_batch("ALTER TABLE memo_generation ADD COLUMN parent INTEGER")
                .map_err(sql)?;
            tx.pragma_update(None, "user_version", DELTA_SCHEMA_VERSION)
                .map_err(sql)?;
        }
        tx.execute(
            "INSERT INTO memo_generation (id, header) VALUES (?1, ?2)",
            params![to_sql(id.0), header],
        )
        .map_err(sql)?;
        if let Some(parent) = parent {
            tx.execute(
                "UPDATE memo_generation SET parent = ?1 WHERE id = ?2",
                params![to_sql(parent.0), to_sql(id.0)],
            )
            .map_err(sql)?;
        }
        let mut items = Vec::with_capacity(tombstones.len());
        {
            let mut insert = tx
                .prepare_cached(
                    "INSERT INTO memo_entry (generation, tbl, key_hash, len, frame) \
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(sql)?;
            for entry in entries {
                let entry = entry?;
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
            for tombstone in tombstones {
                insert
                    .insert(params![
                        to_sql(id.0),
                        tombstone.table,
                        to_sql(tombstone.key_hash),
                        0_u32,
                        &[] as &[u8]
                    ])
                    .map_err(sql)?;
                items.push(IndexItem::tombstone(tombstone.table, tombstone.key_hash));
            }
        }
        tx.execute(
            "INSERT OR REPLACE INTO memo_current (one, generation) VALUES (0, ?1)",
            [to_sql(id.0)],
        )
        .map_err(sql)?;
        tx.commit().map_err(sql)?;
        Ok(items)
    }

    /// Removes every generation outside the current chain. Those below it
    /// are left by a removal that failed or never ran. Those above it, or any
    /// at all with no `memo_current`, only by an outside edit of that row,
    /// since a write commits its rows and the pointer together, and they
    /// would make every later write's id not newer than the newest row.
    fn remove_leftovers(&self) -> Result<()> {
        let mut conn = lock(&self.writer);
        if leftovers(&conn)?.is_empty() {
            return Ok(());
        }
        let ids = leftovers(&conn)?;
        remove_in_one(&mut conn, &ids)
    }
}

fn leftovers(conn: &Connection) -> Result<Vec<GenId>> {
    let chain = chain_ids(conn)?;
    let mut select = conn
        .prepare_cached("SELECT id FROM memo_generation")
        .map_err(sql)?;
    let mut rows = select.query([]).map_err(sql)?;
    let mut ids = Vec::new();
    while let Some(row) = rows.next().map_err(sql)? {
        let id = GenId(from_sql(row.get(0).map_err(sql)?));
        if !chain.contains(&id) {
            ids.push(id);
        }
    }
    Ok(ids)
}

/// Removes the generations' rows and their `memo_generation` rows in one
/// transaction, so a crash leaves them whole for
/// [`SqliteMemo::remove_leftovers`] to finish at the next
/// [`lock_checkpoint`](MemoBackend::lock_checkpoint).
fn remove_in_one(conn: &mut Connection, generations: &[GenId]) -> Result<()> {
    if generations.is_empty() {
        return Ok(());
    }
    let marks = vec!["?"; generations.len()].join(",");
    let ids = || generations.iter().map(|g| to_sql(g.0));
    let tx = match conn.transaction_with_behavior(TransactionBehavior::Immediate) {
        Err(e) if is_busy(&e) => return Err(Error::Busy),
        begun => begun.map_err(sql)?,
    };
    if let Some(live) = chain_ids(&tx)?.iter().find(|id| generations.contains(id)) {
        return Err(Error::Backend(
            format!(
                "memo generation {} is in the current chain and cannot be removed",
                live.0
            )
            .into(),
        ));
    }
    tx.execute(
        &delete_generations_sql(generations.len()),
        params_from_iter(ids()),
    )
    .map_err(sql)?;
    tx.execute(
        &format!("DELETE FROM memo_generation WHERE id IN ({marks})"),
        params_from_iter(ids()),
    )
    .map_err(sql)?;
    tx.commit().map_err(sql)
}

fn create_schema(conn: &mut Connection) -> Result<()> {
    create_schema_for(conn, &[SCHEMA_VERSION, DELTA_SCHEMA_VERSION])
}

/// Creates the schema in an empty database and accepts a memo store whose
/// `user_version` is one of `supported`.
fn create_schema_for(conn: &mut Connection, supported: &[i64]) -> Result<()> {
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
    let objects: i64 = tx
        .query_row("SELECT count(*) FROM sqlite_master", [], |row| row.get(0))
        .map_err(sql)?;
    match (application_id, version) {
        (0, 0) if objects == 0 => {
            tx.execute_batch(SCHEMA).map_err(sql)?;
            tx.pragma_update(None, "application_id", APPLICATION_ID)
                .map_err(sql)?;
            tx.pragma_update(None, "user_version", SCHEMA_VERSION)
                .map_err(sql)?;
        }
        (APPLICATION_ID, version) if supported.contains(&version) => {}
        (0, 0) => {
            return Err(Error::Backend(
                "not an incr-persist memo store: the database holds another schema".into(),
            ))
        }
        _ => {
            return Err(Error::Backend(
                format!(
                    "not an incr-persist memo store of a supported schema version \
                     {supported:?} (application_id {application_id:#x}, user_version {version})"
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

fn user_version(conn: &Connection) -> Result<i64> {
    conn.pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(sql)
}

/// A `memo_generation` row: its header blob, `None` when the row's header
/// column is not a blob, and its parent.
struct Row {
    id: GenId,
    header: Option<Vec<u8>>,
    parent: Option<GenId>,
}

fn read_row(conn: &Connection, with_parent: bool, id: GenId) -> Result<Option<Row>> {
    let sql_text = if with_parent {
        "SELECT header, parent FROM memo_generation WHERE id = ?1"
    } else {
        "SELECT header, NULL FROM memo_generation WHERE id = ?1"
    };
    conn.prepare_cached(sql_text)
        .map_err(sql)?
        .query_row([to_sql(id.0)], |row| {
            Ok(Row {
                id,
                header: row.get(0).ok(),
                parent: row
                    .get::<_, Option<i64>>(1)
                    .ok()
                    .flatten()
                    .map(|p| GenId(from_sql(p))),
            })
        })
        .optional()
        .map_err(sql)
}

/// The rows of the current chain, oldest first, or `None` before any
/// generation was written. A chain that cannot be followed (a parent row
/// missing or a cycle, which only an outside edit makes) is the current
/// row alone, which `layer` then loads as the empty generation it is.
fn chain_rows(conn: &Connection) -> Result<Option<Vec<Row>>> {
    let Some(tip) = current(conn)? else {
        return Ok(None);
    };
    let with_parent = user_version(conn)? >= DELTA_SCHEMA_VERSION;
    let mut rows = Vec::new();
    let mut next = Some(tip);
    while let Some(id) = next {
        let found = if rows.iter().any(|row: &Row| row.id == id) {
            None
        } else {
            read_row(conn, with_parent, id)?
        };
        let Some(found) = found else {
            let alone = read_row(conn, with_parent, tip)?.map(|row| Row {
                parent: None,
                ..row
            });
            return Ok(Some(vec![alone.unwrap_or(Row {
                id: tip,
                header: None,
                parent: None,
            })]));
        };
        next = found.parent;
        rows.push(found);
    }
    rows.reverse();
    Ok(Some(rows))
}

fn chain_ids(conn: &Connection) -> Result<Vec<GenId>> {
    Ok(chain_rows(conn)?
        .map(|rows| rows.iter().map(|row| row.id).collect())
        .unwrap_or_default())
}

enum Head {
    Full(GenHeader),
    Delta(DeltaHeader),
}

/// Loads `row`'s header and index. A full generation whose header or index
/// is unreadable is [`Generation::empty`], a delta, whose `parent` is
/// `Some`, [`Generation::empty_delta`] on `base`.
fn layer(conn: &Connection, row: &Row, base: GenId, parent: Option<GenId>) -> Result<Generation> {
    let blob = row.header.as_deref();
    let empty = |id| match parent {
        Some(parent) => Generation::empty_delta(id, base, parent),
        None => Generation::empty(id),
    };
    let header = match parent {
        None => blob
            .and_then(|blob| parse_header(row.id, blob))
            .map(Head::Full),
        Some(_) => blob
            .and_then(|blob| parse_delta_header(row.id, blob))
            .filter(|delta| delta.base == base && Some(delta.parent) == parent)
            .map(Head::Delta),
    };
    let Some(header) = header else {
        return Ok(empty(row.id));
    };
    let mut select = conn
        .prepare_cached("SELECT loc, tbl, key_hash, len FROM memo_entry WHERE generation = ?1")
        .map_err(sql)?;
    let mut rows = select.query([to_sql(row.id.0)]).map_err(sql)?;
    let mut items = Vec::new();
    while let Some(entry) = rows.next().map_err(sql)? {
        match parse_item(entry) {
            Some(item) => items.push(item),
            None => return Ok(empty(row.id)),
        }
    }
    Ok(match header {
        Head::Full(header) => Generation::new(row.id, header, items),
        Head::Delta(delta) => Generation::new_delta(row.id, delta, items),
    })
}

fn parse_header(id: GenId, blob: &[u8]) -> Option<GenHeader> {
    decode_header(header_frame(id, blob)?).ok()
}

fn parse_delta_header(id: GenId, blob: &[u8]) -> Option<DeltaHeader> {
    decode_delta_header(header_frame(id, blob)?).ok()
}

fn header_frame(id: GenId, blob: &[u8]) -> Option<&[u8]> {
    let (stored, frame) = blob.split_first_chunk::<8>()?;
    (u64::from_le_bytes(*stored) == id.0).then_some(frame)
}

/// `None` when a column holds a value no write could have stored.
fn parse_item(row: &rusqlite::Row<'_>) -> Option<IndexItem> {
    let int = |column| row.get::<_, i64>(column).ok();
    let table = u32::try_from(int(1)?).ok()?;
    let key_hash = from_sql(int(2)?);
    let len = u32::try_from(int(3)?).ok()?;
    if len == 0 {
        return Some(IndexItem::tombstone(table, key_hash));
    }
    Some(IndexItem {
        loc: from_sql(int(0)?),
        table,
        key_hash,
        len,
    })
}

fn to_sql(value: u64) -> i64 {
    value as i64
}

fn from_sql(value: i64) -> u64 {
    value as u64
}

/// Deletes through `memo_entry_index`, which touches every page of the
/// chain, so the checkpoint writes them contiguously. A rowid-ordered
/// delete frees emptied pages without writing them and leaves a scatter of
/// half-emptied ones, whose single `fsync` took 50 to 145 s on a spinning
/// disk against 9 s for this order.
fn delete_generations_sql(generations: usize) -> String {
    let marks = vec!["?"; generations].join(",");
    format!("DELETE FROM memo_entry WHERE generation IN ({marks})")
}

/// Fetches rows by rowid. Without `NOT INDEXED`, a long `IN` list makes
/// SQLite scan the whole generation through `memo_entry_index` for every
/// batch, which is quadratic in the generation's entries.
fn read_entries_sql(locs: usize) -> String {
    let marks = vec!["?"; locs].join(",");
    format!(
        "SELECT loc, frame FROM memo_entry NOT INDEXED WHERE generation = ?1 AND loc IN ({marks})"
    )
}

#[cfg(test)]
#[path = "sqlite_tests.rs"]
mod tests;
