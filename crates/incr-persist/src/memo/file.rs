use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::{
    decode_header, decode_index, encode_header, encode_index, EncodedEntry, GenHeader, GenId,
    Generation, IndexItem, MemoBackend, FRAME_HEADER_LEN,
};
use crate::file::{sync_dir, write_synced};
use crate::{Error, Result};

const CURRENT: &str = "MEMO_CURRENT";
const LOCK: &str = "MEMO_LOCK";
const PUBLISH: &str = "MEMO_PUBLISH";
const GEN_PREFIX: &str = "MEMO_GEN.";
// Not a `.tmp` suffix: a FileBackend log sharing the directory removes
// `*.tmp` files when it opens or compacts.
const TMP_PREFIX: &str = "MEMO_TMP.";
const CURRENT_TMP: &str = "MEMO_TMP.current";
const CURRENT_LEN: usize = 12;
const FOOTER_LEN: usize = 20;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// A memo store kept in a directory, which may be the directory of a
/// [`FileBackend`](crate::FileBackend) log.
///
/// Each generation is one immutable file, `MEMO_GEN.{id}`:
///
/// ```text
/// header frame | entry frames | index frame | footer
/// footer = u64 index offset | u64 generation id | u32 crc32 of those 16 bytes
/// ```
///
/// An [`IndexItem::loc`] is the frame's offset in that file. `MEMO_CURRENT`
/// holds the current id and its crc32. A generation file whose footer names
/// another id, as after files were swapped, is treated as damaged.
///
/// [`write_generation`](MemoBackend::write_generation) writes and fsyncs a
/// temp file, then, under an exclusive lock on `MEMO_PUBLISH`, checks the
/// base, renames the file into place, fsyncs the directory, replaces
/// `MEMO_CURRENT` the same way and fsyncs the directory again.
/// [`lock_checkpoint`](MemoBackend::lock_checkpoint) takes `MEMO_LOCK`
/// without waiting and removes temp files and every generation file but
/// the current one. Locks
/// are taken on a fresh open of the lock file, so they exclude other
/// handles and other threads in this process as well as other processes.
/// That cleanup assumes every writer holds the checkpoint lock: a
/// `write_generation` made without it can lose its temp file and fail
/// with an I/O error.
///
/// A handle keeps the last generation file it read from open. On Unix it
/// therefore keeps reading a generation another process removed after this
/// handle opened it, which returns the frames that generation was written
/// with.
#[derive(Debug)]
pub struct FileMemo {
    dir: PathBuf,
    open: Mutex<Option<(GenId, Arc<File>)>>,
}

/// The checkpoint lock of a [`FileMemo`], released on drop.
#[derive(Debug)]
pub struct FileMemoGuard {
    _lock: File,
}

impl FileMemo {
    /// Opens the memo store in `dir`, creating the directory if needed.
    /// Nothing is read until [`MemoBackend::load`].
    pub fn open(dir: impl AsRef<Path>) -> Result<FileMemo> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(FileMemo {
            dir,
            open: Mutex::new(None),
        })
    }

    fn gen_path(&self, id: GenId) -> PathBuf {
        self.dir.join(format!("{GEN_PREFIX}{}", id.0))
    }

    // Holds only plain data replaced in one assignment, so a panic while it
    // is held leaves nothing half-written.
    fn cache(&self) -> MutexGuard<'_, Option<(GenId, Arc<File>)>> {
        self.open.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The id `MEMO_CURRENT` names, or `None` when no generation was ever
    /// published. A damaged `MEMO_CURRENT` names the highest generation
    /// file present, or `GenId(0)` when there is none.
    fn current(&self) -> Result<Option<GenId>> {
        match fs::read(self.dir.join(CURRENT)) {
            Ok(bytes) => match parse_current(&bytes) {
                Some(id) => Ok(Some(id)),
                None => Ok(Some(GenId(self.highest_generation()?.unwrap_or(0)))),
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn highest_generation(&self) -> Result<Option<u64>> {
        let mut highest = None;
        for entry in fs::read_dir(&self.dir)? {
            if let Some(id) = generation_of(&entry?.file_name()) {
                highest = highest.max(Some(id));
            }
        }
        Ok(highest)
    }

    fn lock_publish(&self) -> Result<File> {
        let file = open_lock_file(&self.dir.join(PUBLISH))?;
        file.lock()?;
        Ok(file)
    }

    /// Removes temp files and every generation file but the current one.
    /// Runs under `MEMO_LOCK` and `MEMO_PUBLISH`, so no checkpoint of any
    /// handle is writing.
    fn remove_leftovers(&self) -> Result<()> {
        let _publish = self.lock_publish()?;
        let current = self.current()?;
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_str().is_some_and(|n| n.starts_with(TMP_PREFIX)) {
                remove_if_present(&entry.path())?;
            } else if let Some(id) = generation_of(&name).map(GenId) {
                if Some(id) != current {
                    self.remove_generation(id)?;
                }
            }
        }
        Ok(())
    }

    /// The open file of generation `id`, or `None` when its file is gone or
    /// is not generation `id`.
    fn generation_file(&self, id: GenId) -> Result<Option<Arc<File>>> {
        if let Some((open_id, file)) = self.cache().as_ref() {
            if *open_id == id {
                return Ok(Some(Arc::clone(file)));
            }
        }
        let file = match File::open(self.gen_path(id)) {
            Ok(file) => file,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let len = file.metadata()?.len();
        if read_footer(&file, len)?.is_none_or(|footer| footer.id != id) {
            return Ok(None);
        }
        let file = Arc::new(file);
        *self.cache() = Some((id, Arc::clone(&file)));
        Ok(Some(file))
    }

    fn publish(&self, base: Option<GenId>, id: GenId, tmp: &Path) -> Result<()> {
        let _publish = self.lock_publish()?;
        if self.current()? != base {
            return Err(Error::Busy);
        }
        fs::rename(tmp, self.gen_path(id))?;
        sync_dir(&self.dir)?;
        let current_tmp = self.dir.join(CURRENT_TMP);
        write_synced(&current_tmp, &current_bytes(id))?;
        fs::rename(&current_tmp, self.dir.join(CURRENT))?;
        sync_dir(&self.dir)
    }
}

impl MemoBackend for FileMemo {
    type Guard = FileMemoGuard;

    fn lock_checkpoint(&self) -> Result<FileMemoGuard> {
        let lock = open_lock_file(&self.dir.join(LOCK))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(Error::Busy),
            Err(TryLockError::Error(e)) => return Err(e.into()),
        }
        self.remove_leftovers()?;
        Ok(FileMemoGuard { _lock: lock })
    }

    fn load(&self) -> Result<Option<Generation>> {
        let Some(mut id) = self.current()? else {
            return Ok(None);
        };
        #[cfg(test)]
        tests::after_load_reads_current();
        // Another handle may publish the next generation and remove this
        // one between reading `MEMO_CURRENT` and opening the file.
        let file = loop {
            if let Some(file) = self.generation_file(id)? {
                break file;
            }
            match self.current()? {
                Some(now) if now != id => id = now,
                _ => return Ok(Some(Generation::empty(id))),
            }
        };
        Ok(Some(
            read_generation(&file, id)?.unwrap_or_else(|| Generation::empty(id)),
        ))
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        let Some(file) = self.generation_file(generation)? else {
            return Ok(None);
        };
        let mut frame = vec![0; item.len as usize];
        let read = read_full_at(&file, &mut frame, item.loc)?;
        frame.truncate(read);
        Ok(Some(frame))
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        let tmp = self.dir.join(format!(
            "{TMP_PREFIX}{}.{}.{}",
            id.0,
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let written = write_generation_file(&tmp, id, header, entries)
            .and_then(|items| self.publish(base, id, &tmp).map(|()| items));
        match written {
            Ok(items) => Ok(Generation::new(id, header.clone(), items)),
            Err(e) => {
                // A temp file that cannot be removed now is removed by the
                // next lock_checkpoint; the write's own error matters more.
                let _ = remove_if_present(&tmp);
                Err(e)
            }
        }
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        {
            let mut cache = self.cache();
            if cache.as_ref().is_some_and(|(id, _)| *id == generation) {
                *cache = None;
            }
        }
        remove_if_present(&self.gen_path(generation))
    }
}

fn open_lock_file(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?)
}

fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e.into()),
        _ => Ok(()),
    }
}

fn generation_of(name: &OsStr) -> Option<u64> {
    name.to_str()?.strip_prefix(GEN_PREFIX)?.parse().ok()
}

fn current_bytes(id: GenId) -> [u8; CURRENT_LEN] {
    let mut bytes = [0; CURRENT_LEN];
    bytes[..8].copy_from_slice(&id.0.to_le_bytes());
    let crc = crc32fast::hash(&bytes[..8]);
    bytes[8..].copy_from_slice(&crc.to_le_bytes());
    bytes
}

fn parse_current(bytes: &[u8]) -> Option<GenId> {
    let bytes: [u8; CURRENT_LEN] = bytes.try_into().ok()?;
    let id = GenId(u64::from_le_bytes(bytes[..8].try_into().ok()?));
    (current_bytes(id) == bytes).then_some(id)
}

/// Streams the generation file to `path`, fsyncs it and returns its index.
fn write_generation_file(
    path: &Path,
    id: GenId,
    header: &GenHeader,
    entries: &[EncodedEntry],
) -> Result<Vec<IndexItem>> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut out = BufWriter::new(file);
    let header = encode_header(header)?;
    out.write_all(&header)?;
    let mut at = header.len() as u64;
    let mut items = Vec::with_capacity(entries.len());
    for entry in entries {
        let len = u32::try_from(entry.frame.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "a memo entry frame exceeds 4 GiB",
            )
        })?;
        items.push(IndexItem {
            table: entry.table,
            key_hash: entry.key_hash,
            loc: at,
            len,
        });
        out.write_all(&entry.frame)?;
        at += u64::from(len);
    }
    out.write_all(&encode_index(&items)?)?;
    out.write_all(&footer_bytes(at, id))?;
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    Ok(items)
}

struct Footer {
    index_at: u64,
    id: GenId,
}

fn footer_bytes(index_at: u64, id: GenId) -> [u8; FOOTER_LEN] {
    let mut bytes = [0; FOOTER_LEN];
    bytes[..8].copy_from_slice(&index_at.to_le_bytes());
    bytes[8..16].copy_from_slice(&id.0.to_le_bytes());
    let crc = crc32fast::hash(&bytes[..16]);
    bytes[16..].copy_from_slice(&crc.to_le_bytes());
    bytes
}

/// The footer of a file `len` bytes long, or `None` when it is damaged.
fn read_footer(file: &File, len: u64) -> Result<Option<Footer>> {
    let Some(at) = len.checked_sub(FOOTER_LEN as u64) else {
        return Ok(None);
    };
    let mut bytes = [0; FOOTER_LEN];
    if read_full_at(file, &mut bytes, at)? < FOOTER_LEN {
        return Ok(None);
    }
    let index_at = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"));
    let id = GenId(u64::from_le_bytes(
        bytes[8..16].try_into().expect("8 bytes"),
    ));
    Ok((footer_bytes(index_at, id) == bytes && index_at <= at).then_some(Footer { index_at, id }))
}

/// The frame at `at`, or `None` when its length word runs past `end`.
fn read_frame(file: &File, at: u64, end: u64) -> Result<Option<Vec<u8>>> {
    let mut word = [0; 4];
    if end.saturating_sub(at) < FRAME_HEADER_LEN as u64 || read_full_at(file, &mut word, at)? < 4 {
        return Ok(None);
    }
    let len = FRAME_HEADER_LEN as u64 + u64::from(u32::from_le_bytes(word));
    if end - at < len {
        return Ok(None);
    }
    let mut frame = vec![0; len as usize];
    Ok((read_full_at(file, &mut frame, at)? == frame.len()).then_some(frame))
}

/// Maps a decode failure to `None`, keeping I/O errors.
fn undamaged<T>(decoded: Result<T>) -> Result<Option<T>> {
    match decoded {
        Ok(value) => Ok(Some(value)),
        Err(Error::Corrupt(_) | Error::Decode(_) | Error::UnsupportedFormat { .. }) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Reads generation `id`'s header and index from `file`, or `None` when
/// either is damaged, the footer names another generation, or an item
/// points outside the entry frames.
fn read_generation(file: &File, id: GenId) -> Result<Option<Generation>> {
    let len = file.metadata()?.len();
    let Some(footer) = read_footer(file, len)? else {
        return Ok(None);
    };
    if footer.id != id {
        return Ok(None);
    }
    let Some(frame) = read_frame(file, 0, footer.index_at)? else {
        return Ok(None);
    };
    let entries_at = frame.len() as u64;
    let Some(header) = undamaged(decode_header(&frame))? else {
        return Ok(None);
    };
    let index_end = len - FOOTER_LEN as u64;
    let Some(frame) = read_frame(file, footer.index_at, index_end)? else {
        return Ok(None);
    };
    if footer.index_at + frame.len() as u64 != index_end {
        return Ok(None);
    }
    let Some(items) = undamaged(decode_index(&frame))? else {
        return Ok(None);
    };
    let in_bounds = items.iter().all(|item| {
        item.loc >= entries_at
            && item
                .loc
                .checked_add(u64::from(item.len))
                .is_some_and(|end| end <= footer.index_at)
    });
    Ok(in_bounds.then(|| Generation::new(id, header, items)))
}

/// Reads into `buf` from `offset` until it is full or the file ends, and
/// returns how many bytes it read.
fn read_full_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let mut done = 0;
    while done < buf.len() {
        match read_at(file, &mut buf[done..], offset + done as u64) {
            Ok(0) => break,
            Ok(read) => done += read,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(_file: &File, _buf: &mut [u8], _offset: u64) -> io::Result<usize> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "FileMemo needs positioned reads",
    ))
}

#[cfg(test)]
#[path = "file_tests.rs"]
mod tests;
