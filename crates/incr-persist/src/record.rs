use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const FORMAT_VERSION: u8 = 1;

/// Whether a declared key holds a single value or a multiset of elements.
// Variant order is part of the wire format: append new variants, never reorder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    /// A single value, changed by [`Record::Set`].
    Input,
    /// A multiset, changed by [`Record::Insert`], [`Record::Remove`] and
    /// [`Record::Replace`].
    Collection,
}

/// One entry in the operation log.
///
/// Every record names a user-chosen string key. Values and elements are
/// serialized bytes that only the typed layer above interprets, so a
/// backend stores and returns them unchanged.
// Variant order is part of the wire format: append new variants, never reorder.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Record {
    /// Introduces `key`. The first `Declare` for a key in log order wins,
    /// and its `default` is what every process resets the key to. For a
    /// collection `default` is empty, since a collection resets to empty.
    Declare {
        key: String,
        kind: Kind,
        default: Vec<u8>,
    },
    /// Sets an input to `value`.
    Set { key: String, value: Vec<u8> },
    /// Adds one occurrence of `element` to a collection.
    Insert { key: String, element: Vec<u8> },
    /// Removes one occurrence of `element` from a collection.
    Remove { key: String, element: Vec<u8> },
    /// Removes one occurrence of `old`, if given, then inserts `new`.
    Replace {
        key: String,
        old: Option<Vec<u8>>,
        new: Vec<u8>,
    },
    /// Resets `key` to its declared default, or empties it if it is a
    /// collection.
    Delete { key: String },
}

impl Record {
    /// The key the record applies to; every variant has one.
    pub fn key(&self) -> &str {
        match self {
            Record::Declare { key, .. }
            | Record::Set { key, .. }
            | Record::Insert { key, .. }
            | Record::Remove { key, .. }
            | Record::Replace { key, .. }
            | Record::Delete { key } => key,
        }
    }

    /// Serializes the record as a format version byte followed by its
    /// `postcard` encoding. The result is self-delimiting only together
    /// with its length, so containers must frame it.
    pub fn encode(&self) -> Result<Vec<u8>> {
        postcard::to_extend(self, vec![FORMAT_VERSION]).map_err(Error::Encode)
    }

    /// Parses bytes produced by [`Record::encode`].
    ///
    /// Fails with [`Error::UnsupportedFormat`] for an unknown version byte,
    /// [`Error::Decode`] when the payload does not deserialize, and
    /// [`Error::Corrupt`] for an empty buffer or bytes left over after the
    /// record.
    pub fn decode(bytes: &[u8]) -> Result<Record> {
        let (&version, payload) = bytes.split_first().ok_or(Error::Corrupt("empty record"))?;
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedFormat {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let (record, rest) = postcard::take_from_bytes(payload).map_err(Error::Decode)?;
        if !rest.is_empty() {
            return Err(Error::Corrupt("trailing bytes after record"));
        }
        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_variant() -> Vec<Record> {
        vec![
            Record::Declare {
                key: "price".into(),
                kind: Kind::Input,
                default: vec![0, 0, 0, 0],
            },
            Record::Declare {
                key: "visits".into(),
                kind: Kind::Collection,
                default: vec![],
            },
            Record::Set {
                key: "price".into(),
                value: vec![1, 2, 3],
            },
            Record::Insert {
                key: "visits".into(),
                element: vec![9; 300],
            },
            Record::Remove {
                key: "visits".into(),
                element: vec![],
            },
            Record::Replace {
                key: "visits".into(),
                old: None,
                new: vec![7],
            },
            Record::Replace {
                key: "visits".into(),
                old: Some(vec![7]),
                new: vec![8],
            },
            Record::Delete {
                key: "naïve/キー".into(),
            },
        ]
    }

    #[test]
    fn every_variant_roundtrips() {
        for record in every_variant() {
            let bytes = record.encode().unwrap();
            assert_eq!(Record::decode(&bytes).unwrap(), record);
        }
    }

    #[test]
    fn encoding_leads_with_the_format_version() {
        for record in every_variant() {
            assert_eq!(record.encode().unwrap()[0], FORMAT_VERSION);
        }
    }

    // Pins the wire format so an accidental reorder of variants or fields
    // fails here instead of misreading logs already on disk.
    #[test]
    fn wire_format_is_stable() {
        let set = Record::Set {
            key: "k".into(),
            value: vec![0xAB],
        };
        assert_eq!(set.encode().unwrap(), [1, 1, 1, b'k', 1, 0xAB]);
        let declare = Record::Declare {
            key: "k".into(),
            kind: Kind::Collection,
            default: vec![],
        };
        assert_eq!(declare.encode().unwrap(), [1, 0, 1, b'k', 1, 0]);
        let delete = Record::Delete { key: "k".into() };
        assert_eq!(delete.encode().unwrap(), [1, 5, 1, b'k']);
    }

    #[test]
    fn unknown_version_is_rejected() {
        let mut bytes = Record::Delete { key: "k".into() }.encode().unwrap();
        bytes[0] = FORMAT_VERSION + 1;
        assert!(matches!(
            Record::decode(&bytes),
            Err(Error::UnsupportedFormat { found, supported })
                if found == FORMAT_VERSION + 1 && supported == FORMAT_VERSION
        ));
    }

    #[test]
    fn empty_buffer_is_corrupt() {
        assert!(matches!(Record::decode(&[]), Err(Error::Corrupt(_))));
    }

    #[test]
    fn trailing_bytes_are_corrupt() {
        let mut bytes = Record::Delete { key: "k".into() }.encode().unwrap();
        bytes.push(0);
        assert!(matches!(Record::decode(&bytes), Err(Error::Corrupt(_))));
    }

    #[test]
    fn truncated_payload_fails_to_decode() {
        let bytes = Record::Set {
            key: "key".into(),
            value: vec![1, 2, 3],
        }
        .encode()
        .unwrap();
        for len in 1..bytes.len() {
            assert!(matches!(
                Record::decode(&bytes[..len]),
                Err(Error::Decode(_))
            ));
        }
    }

    #[test]
    fn key_names_the_record_key() {
        for record in every_variant() {
            let expected = match &record {
                Record::Delete { .. } => "naïve/キー",
                Record::Declare {
                    kind: Kind::Input, ..
                }
                | Record::Set { .. } => "price",
                _ => "visits",
            };
            assert_eq!(record.key(), expected);
        }
    }
}
