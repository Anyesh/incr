use super::*;
use crate::Error;

fn table(name: &str) -> TableDesc {
    TableDesc {
        name: name.into(),
        version: 3,
        key_type: "u32".into(),
        value_type: "alloc::string::String".into(),
    }
}

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
    let frame = encode_entry(&entry()).unwrap();
    assert_eq!(frame_len(&frame), Some(frame.len()));
    assert_eq!(decode_entry(&frame).unwrap(), entry());
}

#[test]
fn every_single_byte_flip_in_an_entry_frame_is_rejected() {
    let frame = encode_entry(&entry()).unwrap();
    for at in 0..frame.len() {
        let mut bad = frame.clone();
        bad[at] ^= 0x10;
        assert!(decode_entry(&bad).is_err(), "flip at byte {at} decoded");
    }
}

#[test]
fn truncated_and_extended_entry_frames_are_rejected() {
    let frame = encode_entry(&entry()).unwrap();
    for len in 0..frame.len() {
        assert!(matches!(
            decode_entry(&frame[..len]),
            Err(Error::Corrupt(_))
        ));
        assert_eq!(frame_len(&frame[..len]), None);
    }
    let mut longer = frame.clone();
    longer.push(0);
    assert!(matches!(decode_entry(&longer), Err(Error::Corrupt(_))));
}

#[test]
fn frames_split_a_concatenation() {
    let a = encode_entry(&entry()).unwrap();
    let b = encode_entry(&Entry {
        last_used: 1,
        ..entry()
    })
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
        decode_header(&encode_entry(&entry()).unwrap()),
        Err(Error::Corrupt(_))
    ));

    let mut body = MAGIC.to_vec();
    body.push(FORMAT + 1);
    body.extend(postcard::to_stdvec(&header).unwrap());
    assert!(matches!(
        decode_header(&frame_bytes(&body)),
        Err(Error::UnsupportedFormat { found, supported }) if found == FORMAT + 1 && supported == FORMAT
    ));
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
