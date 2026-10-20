use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::{Error, Record, Result};

pub(super) const FILE_FORMAT: u8 = 1;
pub(super) const JOURNAL_MAGIC: [u8; 8] = *b"incrjrnl";
pub(super) const SNAPSHOT_MAGIC: [u8; 8] = *b"incrsnap";
pub(super) const HEADER_LEN: usize = 17;

const FRAME_HEADER_LEN: usize = 8;
// Set on every frame of an append except the last, so that readers and
// recovery accept an append only once its final frame is intact.
const CONTINUES: u32 = 1 << 31;
const MAX_FRAME_LEN: usize = (CONTINUES - 1) as usize;

pub(super) const LOCK: &str = "LOCK";
pub(super) const CURRENT: &str = "CURRENT";
const CURRENT_TMP: &str = "CURRENT.tmp";

pub(super) fn journal_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!("journal.{generation}"))
}

pub(super) fn snapshot_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!("snapshot.{generation}"))
}

pub(super) fn snapshot_tmp_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(format!("snapshot.{generation}.tmp"))
}

/// `magic | format version | position (u64 LE)`. For a journal the position
/// is that of its first record; for a snapshot, the position it covers.
pub(super) fn header(magic: [u8; 8], position: u64) -> [u8; HEADER_LEN] {
    let mut bytes = [0; HEADER_LEN];
    bytes[..8].copy_from_slice(&magic);
    bytes[8] = FILE_FORMAT;
    bytes[9..].copy_from_slice(&position.to_le_bytes());
    bytes
}

/// Returns the header's position. Fails with [`Error::Corrupt`] on a short
/// buffer or wrong magic, and [`Error::UnsupportedFormat`] on an unknown
/// version.
pub(super) fn parse_header(bytes: &[u8], magic: [u8; 8]) -> Result<u64> {
    let Some(bytes) = bytes.get(..HEADER_LEN) else {
        return Err(Error::Corrupt("file is shorter than its header"));
    };
    if bytes[..8] != magic {
        return Err(Error::Corrupt("file header has the wrong magic"));
    }
    if bytes[8] != FILE_FORMAT {
        return Err(Error::UnsupportedFormat {
            found: bytes[8],
            supported: FILE_FORMAT,
        });
    }
    Ok(u64::from_le_bytes(bytes[9..].try_into().expect("8 bytes")))
}

/// Reads only the header of the file at `path` and returns its position.
pub(super) fn read_header(path: &Path, magic: [u8; 8]) -> Result<u64> {
    let mut bytes = Vec::with_capacity(HEADER_LEN);
    File::open(path)?
        .take(HEADER_LEN as u64)
        .read_to_end(&mut bytes)?;
    parse_header(&bytes, magic)
}

/// Appends one frame per record, `u32 len | u32 crc32 | record bytes`, and
/// returns each frame's offset relative to where `out` started. The crc
/// covers the length word and the bytes.
pub(super) fn push_frames(out: &mut Vec<u8>, records: &[Record]) -> Result<Vec<usize>> {
    let base = out.len();
    let mut offsets = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let bytes = record.encode()?;
        if bytes.len() > MAX_FRAME_LEN {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "record exceeds the 2 GiB frame limit",
            )));
        }
        let mut word = bytes.len() as u32;
        if index + 1 < records.len() {
            word |= CONTINUES;
        }
        offsets.push(out.len() - base);
        out.extend_from_slice(&word.to_le_bytes());
        out.extend_from_slice(&frame_crc(word, &bytes).to_le_bytes());
        out.extend_from_slice(&bytes);
    }
    Ok(offsets)
}

fn frame_crc(word: u32, bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&word.to_le_bytes());
    hasher.update(bytes);
    hasher.finalize()
}

struct Frame<'a> {
    bytes: &'a [u8],
    continues: bool,
    len: usize,
}

/// Parses the frame at the start of `buf`, or `None` if it is incomplete or
/// fails its crc.
fn frame(buf: &[u8]) -> Option<Frame<'_>> {
    let head = buf.get(..FRAME_HEADER_LEN)?;
    let word = u32::from_le_bytes(head[..4].try_into().expect("4 bytes"));
    let crc = u32::from_le_bytes(head[4..].try_into().expect("4 bytes"));
    let size = (word & !CONTINUES) as usize;
    let bytes = buf.get(FRAME_HEADER_LEN..FRAME_HEADER_LEN.checked_add(size)?)?;
    (frame_crc(word, bytes) == crc).then_some(Frame {
        bytes,
        continues: word & CONTINUES != 0,
        len: FRAME_HEADER_LEN + size,
    })
}

/// The committed prefix of a run of frames.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Scan {
    /// Start offset of every frame in a complete append.
    pub frames: Vec<usize>,
    /// Bytes covered by those frames.
    pub len: usize,
}

/// Walks frames from the start of `buf` and stops at the first frame that
/// is incomplete or fails its crc. Frames of an append whose final frame
/// was not reached are left out.
pub(super) fn scan(buf: &[u8]) -> Scan {
    let mut committed = Scan::default();
    let mut pending = Vec::new();
    let mut at = 0;
    while let Some(frame) = frame(&buf[at..]) {
        pending.push(at);
        at += frame.len;
        if !frame.continues {
            committed.frames.append(&mut pending);
            committed.len = at;
        }
    }
    committed
}

/// Decodes every frame in `buf`, which must consist of whole, valid frames.
pub(super) fn decode_frames(buf: &[u8]) -> Result<Vec<Record>> {
    let mut records = Vec::new();
    let mut at = 0;
    while at < buf.len() {
        let frame = frame(&buf[at..]).ok_or(Error::Corrupt("damaged frame in committed data"))?;
        records.push(Record::decode(frame.bytes)?);
        at += frame.len;
    }
    Ok(records)
}

/// Reads the generation named by `CURRENT`. A missing file surfaces as
/// [`Error::Io`] with [`io::ErrorKind::NotFound`].
pub(super) fn read_current(dir: &Path) -> Result<u64> {
    fs::read_to_string(dir.join(CURRENT))?
        .trim()
        .parse()
        .map_err(|_| Error::Corrupt("CURRENT does not name a generation"))
}

/// Points `CURRENT` at `generation` by writing a temp file, fsyncing it,
/// renaming it over `CURRENT` and fsyncing the directory.
pub(super) fn write_current(dir: &Path, generation: u64) -> Result<()> {
    let tmp = dir.join(CURRENT_TMP);
    write_synced(&tmp, format!("{generation}\n").as_bytes())?;
    fs::rename(&tmp, dir.join(CURRENT))?;
    sync_dir(dir)
}

/// Creates (or truncates) `path`, writes `bytes` and fsyncs, returning the
/// open file for further reads and writes.
pub(crate) fn write_synced(path: &Path, bytes: &[u8]) -> Result<File> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(file)
}

#[cfg(unix)]
pub(crate) fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

// Windows cannot open a directory as a file to fsync it, and NTFS journals
// the rename metadata itself.
#[cfg(not(unix))]
pub(crate) fn sync_dir(_dir: &Path) -> Result<()> {
    Ok(())
}

/// Writes all of `bytes` at `offset` in `file`, as one positioned write
/// where the platform has one rather than a seek and a write. `file` must
/// not be open in append mode, in which Linux ignores the offset.
#[cfg(unix)]
pub(super) fn write_all_at(file: &File, bytes: &[u8], offset: u64) -> io::Result<()> {
    file.write_all_at(bytes, offset)
}

#[cfg(not(unix))]
pub(super) fn write_all_at(mut file: &File, bytes: &[u8], offset: u64) -> io::Result<()> {
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(bytes)
}

/// Removes temp files and every generation's files except `keep`'s.
/// Best-effort because Windows refuses to delete a file another process
/// holds open without share-delete; the next call retries.
pub(super) fn remove_stale(dir: &Path, keep: u64) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let generation = name
            .strip_prefix("journal.")
            .or_else(|| name.strip_prefix("snapshot."))
            .and_then(|rest| rest.parse::<u64>().ok());
        let stale = name.ends_with(".tmp") || generation.is_some_and(|g| g != keep);
        if stale {
            // Ignored on purpose: a file that cannot be removed now is
            // retried by the next compaction or writable open.
            let _ = fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Kind;

    fn records() -> Vec<Record> {
        vec![
            Record::Declare {
                key: "a".into(),
                kind: Kind::Input,
                default: vec![0],
            },
            Record::Set {
                key: "a".into(),
                value: vec![1; 40],
            },
            Record::Delete { key: "a".into() },
        ]
    }

    #[test]
    fn header_roundtrips_and_rejects_damage() {
        let bytes = header(JOURNAL_MAGIC, 42);
        assert_eq!(&bytes[..8], b"incrjrnl");
        assert_eq!(bytes[8], FILE_FORMAT);
        assert_eq!(parse_header(&bytes, JOURNAL_MAGIC).unwrap(), 42);
        assert!(matches!(
            parse_header(&bytes, SNAPSHOT_MAGIC),
            Err(Error::Corrupt(_))
        ));
        assert!(matches!(
            parse_header(&bytes[..HEADER_LEN - 1], JOURNAL_MAGIC),
            Err(Error::Corrupt(_))
        ));
        let mut newer = bytes;
        newer[8] = FILE_FORMAT + 1;
        assert!(matches!(
            parse_header(&newer, JOURNAL_MAGIC),
            Err(Error::UnsupportedFormat { found, .. }) if found == FILE_FORMAT + 1
        ));
    }

    #[test]
    fn frames_roundtrip_and_report_offsets() {
        let mut buf = vec![9, 9];
        let offsets = push_frames(&mut buf, &records()).unwrap();
        assert_eq!(offsets[0], 0);
        let body = &buf[2..];
        let scan = scan(body);
        assert_eq!(scan.frames, offsets);
        assert_eq!(scan.len, body.len());
        assert_eq!(decode_frames(body).unwrap(), records());
    }

    #[test]
    fn scan_drops_an_append_whose_last_frame_is_missing() {
        let mut buf = Vec::new();
        push_frames(&mut buf, &records()[..1]).unwrap();
        let first_len = buf.len();
        let second = push_frames(&mut buf, &records()[1..]).unwrap();
        let cut = first_len + second[1] + 3;
        assert_eq!(
            scan(&buf[..cut]),
            Scan {
                frames: vec![0],
                len: first_len
            }
        );
    }

    #[test]
    fn scan_stops_at_a_crc_mismatch() {
        let mut buf = Vec::new();
        push_frames(&mut buf, &records()[..1]).unwrap();
        let first_len = buf.len();
        push_frames(&mut buf, &records()[1..2]).unwrap();
        push_frames(&mut buf, &records()[2..]).unwrap();
        buf[first_len + FRAME_HEADER_LEN + 2] ^= 0x40;
        assert_eq!(
            scan(&buf),
            Scan {
                frames: vec![0],
                len: first_len
            }
        );
    }

    #[test]
    fn crc_covers_the_continuation_bit() {
        let mut buf = Vec::new();
        push_frames(&mut buf, &records()[..2]).unwrap();
        buf[3] ^= 0x80;
        assert_eq!(scan(&buf), Scan::default());
    }

    #[test]
    fn decode_frames_rejects_partial_input() {
        let mut buf = Vec::new();
        push_frames(&mut buf, &records()).unwrap();
        assert!(matches!(
            decode_frames(&buf[..buf.len() - 1]),
            Err(Error::Corrupt(_))
        ));
    }

    #[test]
    fn current_roundtrips_and_reports_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            read_current(dir.path()),
            Err(Error::Io(e)) if e.kind() == io::ErrorKind::NotFound
        ));
        write_current(dir.path(), 7).unwrap();
        assert_eq!(read_current(dir.path()).unwrap(), 7);
        write_current(dir.path(), 8).unwrap();
        assert_eq!(read_current(dir.path()).unwrap(), 8);
        assert!(!dir.path().join(CURRENT_TMP).exists());
        fs::write(dir.path().join(CURRENT), "x").unwrap();
        assert!(matches!(read_current(dir.path()), Err(Error::Corrupt(_))));
    }

    #[test]
    fn remove_stale_keeps_only_the_named_generation() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "journal.1",
            "snapshot.1",
            "journal.2",
            "snapshot.2",
            "snapshot.3.tmp",
            "journal.3",
            "CURRENT.tmp",
            "CURRENT",
            "LOCK",
            "unrelated",
        ] {
            fs::write(dir.path().join(name), b"").unwrap();
        }
        remove_stale(dir.path(), 2);
        let mut left: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(
            left,
            ["CURRENT", "LOCK", "journal.2", "snapshot.2", "unrelated"]
        );
    }
}
