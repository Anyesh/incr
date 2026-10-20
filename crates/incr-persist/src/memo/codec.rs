//! The byte formats of a memo generation: the header with its table
//! catalog, one frame per entry, and an optional index frame.
//!
//! Every frame is `u32 len | u32 crc32 | body`, all little-endian, and the
//! crc covers the length word and the body, so a flipped or truncated frame
//! is rejected before its body is decoded. Bodies are postcard.

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// The leading bytes of a header frame's body.
pub const MAGIC: [u8; 8] = *b"incrmemo";
/// The format version this build writes and reads, stored after [`MAGIC`].
pub const FORMAT: u8 = 1;
/// The bytes before a frame's body.
pub const FRAME_HEADER_LEN: usize = 8;

/// One memo table as the catalog records it. An entry belongs to a table
/// only while every field matches the live declaration; any difference
/// drops that table's entries and nothing else.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TableDesc {
    pub name: String,
    pub version: u32,
    /// `std::any::type_name` of the key type K.
    pub key_type: String,
    /// `std::any::type_name` of the value type T.
    pub value_type: String,
}

/// What a generation holds besides its entries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenHeader {
    /// The build id's fingerprint. A mismatch drops the whole generation.
    pub build: u128,
    /// The checkpoint sequence number: one more than the generation it
    /// replaced, starting at 1. Entries written live take it as `last_used`.
    pub seq: u64,
    /// Entries and [`StoredDep::Query`] deps name tables by index here.
    pub tables: Vec<TableDesc>,
}

impl GenHeader {
    /// The catalog index of the table that matches `desc` in every field.
    pub fn table_index(&self, desc: &TableDesc) -> Option<u32> {
        self.tables
            .iter()
            .position(|table| table == desc)
            .map(|index| index as u32)
    }
}

/// One dependency of a stored entry with the fingerprint it had when the
/// entry's value was computed. Fingerprints are [`fingerprint`]s.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoredDep {
    /// A Store input: the fingerprint of the bytes its value decoded from.
    Input { key: String, fp: u128 },
    /// A Store collection: the wrapping sum of its elements' fingerprints.
    Collection { key: String, fp: u128 },
    /// A value set through `set_external`, which is its own fingerprint.
    External { name: String, fp: u128 },
    /// Another memo entry: the fingerprint of its encoded value.
    Query { table: u32, key: Vec<u8>, fp: u128 },
}

/// One memoized query result. Its identity within a generation is
/// `(table, key)`; `key` and `value` are the postcard bytes of K and T.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub table: u32,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    /// Every dependency of the run that produced `value`, in read order.
    pub deps: Vec<StoredDep>,
    /// The [`GenHeader::seq`] of the last checkpoint that found this entry
    /// live. Eviction drops the smallest first.
    pub last_used: u64,
}

/// Where a backend keeps one entry frame of a generation, with the
/// identity hash restore looks it up by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexItem {
    pub table: u32,
    /// [`key_hash`] of the entry's key bytes.
    pub key_hash: u64,
    /// A location only the backend that wrote the frame interprets, such
    /// as a file offset or a row id.
    pub loc: u64,
    /// The frame's length in bytes.
    pub len: u32,
}

/// The xxh3-128 every memo fingerprint uses.
pub fn fingerprint(bytes: &[u8]) -> u128 {
    xxhash_rust::xxh3::xxh3_128(bytes)
}

/// The xxh3-64 of an entry's key bytes, which the index is keyed by.
/// Equal hashes are confirmed by comparing the decoded key.
pub fn key_hash(key: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(key)
}

/// The length of the complete frame at the start of `buf`, or `None` when
/// `buf` is shorter than the frame its length word announces. The crc is
/// not checked.
pub fn frame_len(buf: &[u8]) -> Option<usize> {
    let word = buf.get(..4)?;
    let len = FRAME_HEADER_LEN.checked_add(u32::from_le_bytes(word.try_into().ok()?) as usize)?;
    (buf.len() >= len).then_some(len)
}

pub fn encode_entry(entry: &Entry) -> Result<Vec<u8>> {
    Ok(frame_bytes(
        &postcard::to_stdvec(entry).map_err(Error::Encode)?,
    ))
}

/// Decodes one entry from exactly one frame. A crc mismatch, a length that
/// differs from `frame.len()` or an undecodable body is an error.
pub fn decode_entry(frame: &[u8]) -> Result<Entry> {
    decode_body(body(frame)?)
}

pub fn encode_header(header: &GenHeader) -> Result<Vec<u8>> {
    let mut body = MAGIC.to_vec();
    body.push(FORMAT);
    body.extend(postcard::to_stdvec(header).map_err(Error::Encode)?);
    Ok(frame_bytes(&body))
}

/// Decodes a header from exactly one frame, failing with
/// [`Error::UnsupportedFormat`] on another format version.
pub fn decode_header(frame: &[u8]) -> Result<GenHeader> {
    let body = body(frame)?;
    let Some(rest) = body.strip_prefix(&MAGIC) else {
        return Err(Error::Corrupt("memo header has the wrong magic"));
    };
    match rest.split_first() {
        Some((&FORMAT, header)) => decode_body(header),
        Some((&found, _)) => Err(Error::UnsupportedFormat {
            found,
            supported: FORMAT,
        }),
        None => Err(Error::Corrupt("memo header has no format version")),
    }
}

pub fn encode_index(items: &[IndexItem]) -> Result<Vec<u8>> {
    Ok(frame_bytes(
        &postcard::to_stdvec(items).map_err(Error::Encode)?,
    ))
}

pub fn decode_index(frame: &[u8]) -> Result<Vec<IndexItem>> {
    decode_body(body(frame)?)
}

pub(super) fn frame_bytes(body: &[u8]) -> Vec<u8> {
    let word = u32::try_from(body.len()).expect("a memo frame body exceeds 4 GiB");
    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + body.len());
    out.extend_from_slice(&word.to_le_bytes());
    out.extend_from_slice(&crc(word, body).to_le_bytes());
    out.extend_from_slice(body);
    out
}

fn crc(word: u32, body: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&word.to_le_bytes());
    hasher.update(body);
    hasher.finalize()
}

fn body(frame: &[u8]) -> Result<&[u8]> {
    if frame_len(frame) != Some(frame.len()) {
        return Err(Error::Corrupt("memo frame length does not match its bytes"));
    }
    let word = u32::from_le_bytes(frame[..4].try_into().expect("4 bytes"));
    let stored = u32::from_le_bytes(frame[4..8].try_into().expect("4 bytes"));
    let body = &frame[FRAME_HEADER_LEN..];
    if crc(word, body) != stored {
        return Err(Error::Corrupt("memo frame fails its crc"));
    }
    Ok(body)
}

fn decode_body<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T> {
    let (value, rest) = postcard::take_from_bytes(body).map_err(Error::Decode)?;
    if !rest.is_empty() {
        return Err(Error::Corrupt("memo frame has trailing bytes"));
    }
    Ok(value)
}

#[cfg(test)]
#[path = "codec_tests.rs"]
mod tests;
