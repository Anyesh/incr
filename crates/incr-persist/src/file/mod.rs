use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use crate::{Backend, Error, Position, ReadOutcome, Record, Result, SnapshotId};

use format::{
    decode_frames, header, journal_path, parse_header, push_frames, read_current, read_header,
    remove_stale, scan, snapshot_path, snapshot_tmp_path, write_all_at, write_current, HEADER_LEN,
    JOURNAL_MAGIC, LOCK, SNAPSHOT_MAGIC,
};
pub(crate) use format::{sync_dir, write_synced};

mod format;
#[cfg(test)]
mod tests;

const COMPACTION_STEPS: u8 = 5;

/// A log kept in a directory, with one writer and any number of readers.
///
/// The directory holds `journal.N` (records after the snapshot),
/// `snapshot.N` (absent for generation 0, which was never compacted),
/// `CURRENT` naming the live generation `N`, and `LOCK`. Each data file
/// starts with a header carrying the file format version and a position;
/// records follow as `u32 len | u32 crc32 | bytes` frames. The top bit of
/// `len` marks every frame of an append except its last, so readers and
/// crash recovery see an append whole or not at all.
///
/// [`FileBackend::open`] takes an exclusive lock on `LOCK` for the
/// handle's lifetime, so at most one writable handle exists per log across
/// all processes. Read-only handles take no lock and follow the writer,
/// including across compactions: a reader keeps reading a journal that
/// compaction unlinked through its open file, and moves on to the new
/// journal within the same read once it has read everything the snapshot
/// covers. A reader that fell behind more than one compaction gets
/// [`ReadOutcome::Compacted`] instead.
///
/// `append` writes without fsync; [`Backend::sync`] fsyncs the journal.
#[derive(Debug)]
pub struct FileBackend {
    dir: PathBuf,
    state: Mutex<State>,
    // Held for the handle's lifetime so that no second writer can open the
    // log; dropping the file releases the lock. `None` for read-only handles.
    lock: Option<File>,
}

#[derive(Debug)]
struct State {
    generation: u64,
    journal: Journal,
    synced: u64,
    // Set when a failed write may have left the directory out of step with
    // this handle, which must then refuse writes until the log is reopened.
    failed: bool,
}

#[derive(Debug)]
struct Journal {
    file: File,
    start: u64,
    offsets: Vec<u64>,
    end: u64,
}

impl Journal {
    /// Opens and indexes `journal.{generation}`. A writable open also cuts
    /// off a torn or uncommitted tail and fsyncs what remains.
    fn open(dir: &Path, generation: u64, writable: bool) -> Result<Journal> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(writable)
            .open(journal_path(dir, generation))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let start = parse_header(&bytes, JOURNAL_MAGIC)?;
        if start == 0 {
            return Err(Error::Corrupt("journal starts at position zero"));
        }
        let mut journal = Journal {
            file,
            start,
            offsets: Vec::new(),
            end: HEADER_LEN as u64,
        };
        journal.index(&bytes[HEADER_LEN..]);
        if writable {
            if journal.end < bytes.len() as u64 {
                journal.file.set_len(journal.end)?;
            }
            journal.file.sync_data()?;
        }
        Ok(journal)
    }

    fn index(&mut self, from_end: &[u8]) {
        let scan = scan(from_end);
        let end = self.end;
        self.offsets
            .extend(scan.frames.iter().map(|&at| end + at as u64));
        self.end += scan.len as u64;
    }

    fn catch_up(&mut self) -> Result<()> {
        let mut buf = Vec::new();
        self.file.seek(SeekFrom::Start(self.end))?;
        self.file.read_to_end(&mut buf)?;
        self.index(&buf);
        Ok(())
    }

    fn snapshot_at(&self) -> u64 {
        self.start - 1
    }

    fn tail(&self) -> u64 {
        self.start - 1 + self.offsets.len() as u64
    }

    fn offset_after(&self, after: u64) -> u64 {
        let index = (after + 1 - self.start) as usize;
        self.offsets.get(index).copied().unwrap_or(self.end)
    }

    /// The committed frames for every position above `after`, which must
    /// lie between the snapshot position and the tail.
    fn frames_after(&mut self, after: u64) -> Result<Vec<u8>> {
        let from = self.offset_after(after);
        let mut buf = vec![0; (self.end - from) as usize];
        self.file.seek(SeekFrom::Start(from))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Pushes every committed record above `after`, which must be at or
    /// above the snapshot position, and returns the last position read.
    fn read_to_tail(&mut self, after: u64, out: &mut Vec<(Position, Record)>) -> Result<u64> {
        let tail = self.tail();
        if after >= tail {
            return Ok(after);
        }
        let records = decode_frames(&self.frames_after(after)?)?;
        out.extend((after + 1..).map(Position).zip(records));
        Ok(tail)
    }
}

/// Opens the journal `CURRENT` names, starting from `generation`. A journal
/// that vanished because a compaction raced the open is retried at the
/// generation `CURRENT` names now; one that is missing while `CURRENT`
/// still names it is an error.
fn open_latest(dir: &Path, mut generation: u64) -> Result<(u64, Journal)> {
    loop {
        match Journal::open(dir, generation, false) {
            Err(Error::Io(e)) if e.kind() == io::ErrorKind::NotFound => {
                let now = read_current(dir)?;
                if now == generation {
                    return Err(Error::Io(e));
                }
                generation = now;
            }
            opened => return opened.map(|journal| (generation, journal)),
        }
    }
}

fn not_open_for_writing() -> Error {
    Error::Io(io::Error::other(
        "an earlier failed write left this handle out of step with the log; reopen it",
    ))
}

impl FileBackend {
    /// Opens the log in `dir` for writing, creating the directory and an
    /// empty log if there is none.
    ///
    /// Fails with an [`Error::Io`] of kind
    /// [`ResourceBusy`](io::ErrorKind::ResourceBusy) while another handle,
    /// in this process or another, has the log open for writing. Truncates a
    /// torn tail left by a crash and removes files left behind by an
    /// interrupted compaction. Fails with [`Error::UnsupportedFormat`] or
    /// [`Error::Corrupt`] on files it cannot read.
    pub fn open(dir: impl AsRef<Path>) -> Result<FileBackend> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join(LOCK))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::ResourceBusy,
                    format!(
                        "log at {} is already open for writing by another handle or process",
                        dir.display()
                    ),
                )))
            }
            Err(TryLockError::Error(e)) => return Err(e.into()),
        }
        let generation = match read_current(&dir) {
            Ok(generation) => generation,
            Err(Error::Io(e)) if e.kind() == io::ErrorKind::NotFound => {
                write_synced(&journal_path(&dir, 0), &header(JOURNAL_MAGIC, 1))?;
                write_current(&dir, 0)?;
                0
            }
            Err(e) => return Err(e),
        };
        remove_stale(&dir, generation);
        let journal = Journal::open(&dir, generation, true)?;
        let synced = journal.tail();
        Ok(FileBackend {
            dir,
            state: Mutex::new(State {
                generation,
                journal,
                synced,
                failed: false,
            }),
            lock: Some(lock),
        })
    }

    /// Opens an existing log in `dir` for reading only. Takes no lock, so
    /// it works while a writer has the log open, and never modifies the
    /// directory; a torn tail is skipped rather than truncated.
    ///
    /// Fails with an [`Error::Io`] of kind
    /// [`NotFound`](io::ErrorKind::NotFound) when `dir` holds no log.
    pub fn open_read_only(dir: impl AsRef<Path>) -> Result<FileBackend> {
        let dir = dir.as_ref().to_path_buf();
        let generation = match read_current(&dir) {
            Err(Error::Io(e)) if e.kind() == io::ErrorKind::NotFound => {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("no log at {}: CURRENT is missing", dir.display()),
                )))
            }
            current => current?,
        };
        let (generation, journal) = open_latest(&dir, generation)?;
        Ok(FileBackend {
            dir,
            state: Mutex::new(State {
                generation,
                journal,
                synced: 0,
                failed: false,
            }),
            lock: None,
        })
    }

    fn state(&self) -> Result<MutexGuard<'_, State>> {
        self.state.lock().map_err(|_| {
            Error::Io(io::Error::other(
                "a call on this handle panicked mid-update; reopen the log",
            ))
        })
    }

    fn writer_state(&self) -> Result<MutexGuard<'_, State>> {
        if self.lock.is_none() {
            return Err(Error::ReadOnly);
        }
        let state = self.state()?;
        if state.failed {
            return Err(not_open_for_writing());
        }
        Ok(state)
    }

    /// Moves a read-only handle to the generation `CURRENT` names, if that
    /// differs from the one it reads. Returns whether it moved.
    fn switch_to_latest(&self, state: &mut State) -> Result<bool> {
        let generation = read_current(&self.dir)?;
        if generation == state.generation {
            return Ok(false);
        }
        let (generation, journal) = open_latest(&self.dir, generation)?;
        state.generation = generation;
        state.journal = journal;
        Ok(true)
    }

    /// Runs compaction through step `stop_after` of five and returns there,
    /// which lets tests simulate a crash between steps:
    ///
    /// 1. write `snapshot.N+1.tmp` and fsync it;
    /// 2. rename it to `snapshot.N+1`;
    /// 3. write `journal.N+1` with the records above `upto` and fsync it;
    /// 4. point `CURRENT` at `N+1` by atomic rename and fsync the directory;
    /// 5. switch this handle over and delete the old pair, best-effort.
    fn compact_steps(&self, upto: Position, folded: Vec<Record>, stop_after: u8) -> Result<()> {
        let mut guard = self.writer_state()?;
        let state = &mut *guard;
        let (snapshot_at, tail) = (state.journal.snapshot_at(), state.journal.tail());
        if upto.0 < snapshot_at || upto.0 > tail {
            return Err(Error::InvalidCompaction {
                upto,
                snapshot_at: Position(snapshot_at),
                tail: Position(tail),
            });
        }
        let dir = &self.dir;
        let next = state.generation + 1;

        let mut snapshot = header(SNAPSHOT_MAGIC, upto.0).to_vec();
        push_frames(&mut snapshot, &folded)?;
        let tmp = snapshot_tmp_path(dir, next);
        write_synced(&tmp, &snapshot)?;
        if stop_after == 1 {
            return Ok(());
        }

        fs::rename(&tmp, snapshot_path(dir, next))?;
        if stop_after == 2 {
            return Ok(());
        }

        let kept_from = state.journal.offset_after(upto.0);
        let mut journal = header(JOURNAL_MAGIC, upto.0 + 1).to_vec();
        journal.extend_from_slice(&state.journal.frames_after(upto.0)?);
        let file = write_synced(&journal_path(dir, next), &journal)?;
        if stop_after == 3 {
            return Ok(());
        }

        if let Err(e) = write_current(dir, next) {
            // The rename may have landed even though the call failed, so
            // this handle can no longer tell which journal is live.
            state.failed = true;
            return Err(e);
        }
        if stop_after == 4 {
            return Ok(());
        }

        let offsets = state
            .journal
            .offsets
            .iter()
            .filter(|&&at| at >= kept_from)
            .map(|&at| HEADER_LEN as u64 + at - kept_from)
            .collect();
        state.journal = Journal {
            file,
            start: upto.0 + 1,
            offsets,
            end: journal.len() as u64,
        };
        state.generation = next;
        state.synced = state.journal.tail();
        remove_stale(dir, next);
        Ok(())
    }
}

impl Backend for FileBackend {
    fn append(&self, records: &[Record]) -> Result<Position> {
        let mut guard = self.writer_state()?;
        let state = &mut *guard;
        if records.is_empty() {
            return Ok(Position(state.journal.tail()));
        }
        let mut frames = Vec::new();
        let offsets = push_frames(&mut frames, records)?;
        let journal = &mut state.journal;
        let at = journal.end;
        if let Err(e) = write_all_at(&journal.file, &frames, at) {
            // Readers never index an incomplete append, but a partial one
            // must not stay on disk where the next append would land.
            if journal.file.set_len(at).is_err() {
                state.failed = true;
            }
            return Err(e.into());
        }
        journal
            .offsets
            .extend(offsets.iter().map(|&offset| at + offset as u64));
        journal.end += frames.len() as u64;
        Ok(Position(journal.tail()))
    }

    /// Fsyncs the journal. Fails with [`Error::ReadOnly`] on a read-only
    /// handle, which has written nothing it could make durable.
    fn sync(&self, upto: Position) -> Result<()> {
        if self.lock.is_none() {
            return Err(Error::ReadOnly);
        }
        let mut state = self.state()?;
        let target = upto.0.min(state.journal.tail());
        if state.synced < target {
            state.journal.file.sync_data()?;
            state.synced = state.journal.tail();
        }
        Ok(())
    }

    fn read_from(&self, after: Position, out: &mut Vec<(Position, Record)>) -> Result<ReadOutcome> {
        let mut after = after.0;
        let pushed = out.len();
        let writable = self.writable();
        let mut guard = self.state()?;
        let state = &mut *guard;
        loop {
            if !writable {
                state.journal.catch_up()?;
            }
            if after < state.journal.snapshot_at() {
                if !writable && self.switch_to_latest(state)? {
                    continue;
                }
                // Whatever this call drained from an older journal is folded
                // into the snapshot too, and the contract pushes nothing
                // alongside Compacted.
                out.truncate(pushed);
                return Ok(ReadOutcome::Compacted {
                    snapshot_at: Position(state.journal.snapshot_at()),
                });
            }
            after = state.journal.read_to_tail(after, out)?;
            if writable || read_current(&self.dir)? == state.generation {
                return Ok(ReadOutcome::Records);
            }
            // CURRENT has moved on, so the writer appends nothing more to
            // this journal: drain it, then carry on in the journal CURRENT
            // names, so a reader that was behind when compaction ran
            // continues instead of reloading, and one read still reaches the
            // tail.
            state.journal.catch_up()?;
            after = state.journal.read_to_tail(after, out)?;
            self.switch_to_latest(state)?;
        }
    }

    /// The generation is the `N` that `CURRENT` names, since every
    /// compaction moves it on by one. A writer answers from memory, as no
    /// other handle can compact; a reader reads `CURRENT` and, when it names
    /// a generation other than the journal it holds, that snapshot's header.
    fn snapshot_id(&self) -> Result<Option<SnapshotId>> {
        let state = self.state()?;
        let held = |generation: u64| SnapshotId {
            position: Position(state.journal.snapshot_at()),
            generation,
        };
        if self.writable() && !state.failed {
            return Ok((state.generation > 0).then(|| held(state.generation)));
        }
        let mut generation = read_current(&self.dir)?;
        loop {
            if generation == 0 {
                return Ok(None);
            }
            if generation == state.generation {
                return Ok(Some(held(generation)));
            }
            match read_header(&snapshot_path(&self.dir, generation), SNAPSHOT_MAGIC) {
                Ok(at) => {
                    return Ok(Some(SnapshotId {
                        position: Position(at),
                        generation,
                    }))
                }
                Err(Error::Io(e)) if e.kind() == io::ErrorKind::NotFound => {
                    let now = read_current(&self.dir)?;
                    if now == generation {
                        return Err(Error::Io(e));
                    }
                    generation = now;
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn load_snapshot(&self) -> Result<Option<(Position, Vec<Record>)>> {
        let mut generation = read_current(&self.dir)?;
        loop {
            if generation == 0 {
                return Ok(None);
            }
            match fs::read(snapshot_path(&self.dir, generation)) {
                Ok(bytes) => {
                    let at = parse_header(&bytes, SNAPSHOT_MAGIC)?;
                    let records = decode_frames(&bytes[HEADER_LEN..])?;
                    return Ok(Some((Position(at), records)));
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    let now = read_current(&self.dir)?;
                    if now == generation {
                        return Err(e.into());
                    }
                    generation = now;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> Result<()> {
        self.compact_steps(upto, folded, COMPACTION_STEPS)
    }

    fn writable(&self) -> bool {
        self.lock.is_some()
    }
}
