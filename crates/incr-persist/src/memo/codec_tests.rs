use super::*;
use crate::memo::GenId;
use crate::Error;

fn table(name: &str) -> TableDesc {
    TableDesc {
        name: name.into(),
        version: 3,
        key_type: "u32".into(),
        value_type: "alloc::string::String".into(),
    }
}

const TAG: u64 = 0x7a6;

fn entry() -> Entry {
    Entry {
        table: 1,
        key: vec![7, 0, 7],
        value: b"value".to_vec(),
        deps: vec![
            StoredDep::Input {
                key: "a".into(),
                fp: 1,
            },
            StoredDep::Collection {
                key: "c".into(),
                fp: u128::MAX,
            },
            StoredDep::External {
                name: "clock".into(),
                fp: 2,
            },
            StoredDep::Query {
                table: 0,
                key: vec![1],
                fp: 3,
            },
        ],
        last_used: 9,
    }
}

#[test]
fn entry_roundtrips_through_its_frame() {
    let frame = encode_entry(&entry(), TAG).unwrap();
    assert_eq!(frame_len(&frame), Some(frame.len()));
    assert_eq!(decode_entry(&frame, TAG).unwrap(), entry());
}

#[test]
fn an_entry_frame_of_another_generation_is_rejected() {
    let frame = encode_entry(&entry(), TAG).unwrap();
    assert!(matches!(
        decode_entry(&frame, TAG + 1),
        Err(Error::Corrupt(_))
    ));
}

#[test]
fn the_frame_tag_tells_headers_apart() {
    let header = GenHeader {
        build: 1,
        seq: 2,
        tables: vec![table("a"), table("b")],
    };
    assert_eq!(header.frame_tag(), header.clone().frame_tag());
    let reordered = GenHeader {
        tables: vec![table("b"), table("a")],
        ..header.clone()
    };
    let next = GenHeader {
        seq: 3,
        ..header.clone()
    };
    let rebuilt = GenHeader {
        build: 2,
        ..header.clone()
    };
    for other in [reordered, next, rebuilt] {
        assert_ne!(header.frame_tag(), other.frame_tag(), "{other:?}");
    }
}

#[test]
fn every_single_byte_flip_in_an_entry_frame_is_rejected() {
    let frame = encode_entry(&entry(), TAG).unwrap();
    for at in 0..frame.len() {
        let mut bad = frame.clone();
        bad[at] ^= 0x10;
        assert!(
            decode_entry(&bad, TAG).is_err(),
            "flip at byte {at} decoded"
        );
    }
}

#[test]
fn truncated_and_extended_entry_frames_are_rejected() {
    let frame = encode_entry(&entry(), TAG).unwrap();
    for len in 0..frame.len() {
        assert!(matches!(
            decode_entry(&frame[..len], TAG),
            Err(Error::Corrupt(_))
        ));
        assert_eq!(frame_len(&frame[..len]), None);
    }
    let mut longer = frame.clone();
    longer.push(0);
    assert!(matches!(decode_entry(&longer, TAG), Err(Error::Corrupt(_))));
}

#[test]
fn frames_split_a_concatenation() {
    let a = encode_entry(&entry(), TAG).unwrap();
    let b = encode_entry(
        &Entry {
            last_used: 1,
            ..entry()
        },
        TAG,
    )
    .unwrap();
    let joined = [a.clone(), b.clone()].concat();
    let first = frame_len(&joined).unwrap();
    assert_eq!(&joined[..first], &a[..]);
    assert_eq!(frame_len(&joined[first..]), Some(b.len()));
}

#[test]
fn header_roundtrips_and_checks_magic_and_format() {
    let header = GenHeader {
        build: 0xdead_beef_u128 << 64,
        seq: 4,
        tables: vec![table("a"), table("b")],
    };
    let frame = encode_header(&header).unwrap();
    assert_eq!(decode_header(&frame).unwrap(), header);

    let mut bad = frame.clone();
    bad[FRAME_HEADER_LEN] ^= 1;
    assert!(decode_header(&bad).is_err());

    assert!(matches!(
        decode_header(&encode_entry(&entry(), TAG).unwrap()),
        Err(Error::Corrupt(_))
    ));

    // Format 1 entry frames carry no frame tag, and a format 2 build
    // cannot decode `StoredDep::Row`.
    for other in [1, 2, FORMAT + 1] {
        let mut body = MAGIC.to_vec();
        body.push(other);
        body.extend(postcard::to_stdvec(&header).unwrap());
        assert!(matches!(
            decode_header(&frame_bytes(&body).unwrap()),
            Err(Error::UnsupportedFormat { found, supported }) if found == other && supported == FORMAT
        ));
    }
}

#[test]
fn index_roundtrips() {
    let items = vec![
        IndexItem {
            table: 0,
            key_hash: 5,
            loc: 17,
            len: 40,
        },
        IndexItem {
            table: 2,
            key_hash: u64::MAX,
            loc: u64::MAX,
            len: 1,
        },
    ];
    let frame = encode_index(&items).unwrap();
    assert_eq!(decode_index(&frame).unwrap(), items);
    let mut bad = frame.clone();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert!(decode_index(&bad).is_err());
}

#[test]
fn table_index_requires_every_field_to_match() {
    let header = GenHeader {
        build: 1,
        seq: 1,
        tables: vec![table("a"), table("b")],
    };
    assert_eq!(header.table_index(&table("b")), Some(1));
    assert_eq!(header.table_index(&table("c")), None);
    for changed in [
        TableDesc {
            version: 4,
            ..table("a")
        },
        TableDesc {
            key_type: "u64".into(),
            ..table("a")
        },
        TableDesc {
            value_type: "f64".into(),
            ..table("a")
        },
    ] {
        assert_eq!(header.table_index(&changed), None);
    }
}

#[test]
fn hashes_are_xxh3() {
    assert_eq!(fingerprint(b"abc"), xxhash_rust::xxh3::xxh3_128(b"abc"));
    assert_eq!(key_hash(b"abc"), xxhash_rust::xxh3::xxh3_64(b"abc"));
    assert_ne!(fingerprint(b"abc"), fingerprint(b"abd"));
}

#[test]
fn a_frame_length_must_fit_a_u32() {
    let largest = u32::MAX as usize - FRAME_HEADER_LEN;
    assert_eq!(frame_word(largest).unwrap(), u32::MAX - 8);
    assert!(matches!(
        frame_word(largest + 1),
        Err(Error::FrameTooLarge { len }) if len == u64::from(u32::MAX) + 1
    ));
    assert!(matches!(
        frame_word(usize::MAX),
        Err(Error::FrameTooLarge { len: u64::MAX })
    ));
}

#[test]
fn full_generation_bytes_are_unchanged() {
    let header = GenHeader {
        build: 0x0123_4567_89ab_cdef_u128 << 32,
        seq: 300,
        tables: vec![table("a"), table("b")],
    };
    let golden_header: &[u8] = &[
        83, 0, 0, 0, 248, 163, 238, 237, 105, 110, 99, 114, 109, 101, 109, 111, 3, 128, 128, 128,
        128, 240, 189, 243, 213, 137, 207, 149, 154, 18, 172, 2, 2, 1, 97, 3, 3, 117, 51, 50, 21,
        97, 108, 108, 111, 99, 58, 58, 115, 116, 114, 105, 110, 103, 58, 58, 83, 116, 114, 105,
        110, 103, 1, 98, 3, 3, 117, 51, 50, 21, 97, 108, 108, 111, 99, 58, 58, 115, 116, 114, 105,
        110, 103, 58, 58, 83, 116, 114, 105, 110, 103,
    ];
    let golden_entry: &[u8] = &[
        60, 0, 0, 0, 9, 172, 206, 252, 10, 137, 106, 140, 3, 85, 235, 29, 1, 3, 7, 0, 7, 5, 118,
        97, 108, 117, 101, 4, 0, 1, 97, 1, 1, 1, 99, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 3, 2, 5, 99, 108, 111, 99, 107, 2, 3, 0, 1, 1,
        3, 9,
    ];
    assert_eq!(encode_header(&header).unwrap(), golden_header);
    assert_eq!(
        encode_entry(&entry(), header.frame_tag()).unwrap(),
        golden_entry
    );
}

fn delta_header() -> DeltaHeader {
    DeltaHeader {
        header: GenHeader {
            build: 5,
            seq: 12,
            tables: vec![table("a"), table("b")],
        },
        base: GenId(9),
        parent: GenId(11),
    }
}

#[test]
fn delta_header_roundtrips_as_format_4() {
    let delta = delta_header();
    let frame = encode_delta_header(&delta).unwrap();
    assert_eq!(frame[FRAME_HEADER_LEN + MAGIC.len()], DELTA_FORMAT);
    assert_eq!(DELTA_FORMAT, 4);
    assert_eq!(decode_delta_header(&frame).unwrap(), delta);

    let mut bad = frame.clone();
    bad[FRAME_HEADER_LEN + 3] ^= 1;
    assert!(matches!(decode_delta_header(&bad), Err(Error::Corrupt(_))));

    assert!(matches!(
        decode_header(&frame),
        Err(Error::UnsupportedFormat {
            found: 4,
            supported: FORMAT
        })
    ));
    let full = encode_header(&delta.header).unwrap();
    assert!(matches!(decode_delta_header(&full), Err(Error::Corrupt(_))));
}

#[test]
fn format_5_is_rejected_by_both_header_decoders() {
    let delta = delta_header();
    let mut full = MAGIC.to_vec();
    full.push(5);
    full.extend(postcard::to_stdvec(&delta.header).unwrap());
    assert!(matches!(
        decode_header(&frame_bytes(&full).unwrap()),
        Err(Error::UnsupportedFormat {
            found: 5,
            supported: FORMAT
        })
    ));
    let mut body = MAGIC.to_vec();
    body.push(5);
    body.extend(postcard::to_stdvec(&delta).unwrap());
    assert!(matches!(
        decode_delta_header(&frame_bytes(&body).unwrap()),
        Err(Error::UnsupportedFormat {
            found: 5,
            supported: DELTA_FORMAT
        })
    ));
}

#[test]
fn a_delta_tag_covers_its_header_base_and_parent() {
    let delta = delta_header();
    let bytes = postcard::to_stdvec(&delta).unwrap();
    assert_eq!(delta.frame_tag(), xxhash_rust::xxh3::xxh3_64(&bytes));
    assert_ne!(delta.frame_tag(), delta.header.frame_tag());
    let rebased = DeltaHeader {
        base: GenId(8),
        ..delta.clone()
    };
    let reparented = DeltaHeader {
        parent: GenId(10),
        ..delta.clone()
    };
    let next = DeltaHeader {
        header: GenHeader {
            seq: 13,
            ..delta.header.clone()
        },
        ..delta.clone()
    };
    for other in [rebased, reparented, next] {
        assert_ne!(delta.frame_tag(), other.frame_tag(), "{other:?}");
    }
}

#[test]
fn retag_frame_keeps_the_value_and_validates_under_the_new_tag() {
    let delta = delta_header();
    let old = delta.header.frame_tag();
    let new = delta.frame_tag();
    let frame = encode_entry(&entry(), old).unwrap();
    let retagged = retag_frame(&frame, old, new).unwrap();
    assert_eq!(retagged.table, entry().table);
    assert_eq!(retagged.key_hash, key_hash(&entry().key));
    assert_eq!(retagged.frame, encode_entry(&entry(), new).unwrap());
    assert_eq!(retagged.frame.len(), frame.len());
    assert_eq!(retagged.frame[16..], frame[16..]);
    assert_eq!(decode_entry(&retagged.frame, new).unwrap(), entry());
    assert!(matches!(
        decode_entry(&retagged.frame, old),
        Err(Error::Corrupt(_))
    ));
}

#[test]
fn retag_frame_rejects_a_bad_crc_tag_or_body() {
    let frame = encode_entry(&entry(), TAG).unwrap();
    assert!(matches!(
        retag_frame(&frame, TAG + 1, 1),
        Err(Error::Corrupt(_))
    ));
    for at in [2, 5, 9, 17, frame.len() - 1] {
        let mut bad = frame.clone();
        bad[at] ^= 0x40;
        assert!(retag_frame(&bad, TAG, 1).is_err(), "flip at {at}");
    }
    assert!(retag_frame(&frame[..frame.len() - 1], TAG, 1).is_err());
    let mut tag_only = TAG.to_le_bytes().to_vec();
    tag_only.push(0x80);
    assert!(retag_frame(&frame_bytes(&tag_only).unwrap(), TAG, 1).is_err());
}

#[test]
fn a_tombstone_is_an_item_of_length_zero() {
    let tombstone = IndexItem::tombstone(3, 0xfeed);
    assert_eq!(
        tombstone,
        IndexItem {
            table: 3,
            key_hash: 0xfeed,
            loc: 0,
            len: 0,
        }
    );
    assert!(tombstone.is_tombstone());
    let frame = encode_entry(&entry(), TAG).unwrap();
    let live = IndexItem {
        len: frame.len() as u32,
        ..tombstone
    };
    assert!(!live.is_tombstone());
    let items = vec![tombstone, live];
    assert_eq!(decode_index(&encode_index(&items).unwrap()).unwrap(), items);
}
