use super::*;
use crate::memo::{encode_entry, key_hash, EncodedEntry, Entry, GenHeader, IndexItem};

fn item(table: u32, key_hash: u64, loc: u64) -> IndexItem {
    IndexItem {
        table,
        key_hash,
        loc,
        len: 10 + loc as u32,
    }
}

fn header() -> GenHeader {
    GenHeader {
        build: 1,
        seq: 1,
        tables: Vec::new(),
    }
}

#[test]
fn find_returns_every_item_with_the_table_and_hash() {
    let generation = Generation::new(
        GenId(3),
        header(),
        vec![
            item(1, 9, 0),
            item(0, 9, 1),
            item(1, 4, 2),
            item(1, 9, 3),
            item(2, 9, 4),
        ],
    );
    let mut found: Vec<_> = generation.find(1, 9).iter().map(|i| i.loc).collect();
    found.sort();
    assert_eq!(found, [0, 3]);
    assert_eq!(generation.find(0, 9).len(), 1);
    assert!(generation.find(0, 4).is_empty());
    assert!(generation.find(3, 9).is_empty());
    assert_eq!(generation.items().len(), 5);
    assert_eq!(generation.entry_bytes(), 50 + 10);
}

#[test]
fn empty_generation_holds_nothing_and_matches_no_build() {
    let generation = Generation::empty(GenId(4));
    assert_eq!(generation.id(), GenId(4));
    assert!(generation.items().is_empty());
    assert!(generation.header().tables.is_empty());
}

#[test]
fn encoded_entry_carries_the_frame_and_the_key_hash() {
    let entry = Entry {
        table: 2,
        key: vec![1, 2, 3],
        value: vec![4],
        deps: Vec::new(),
        last_used: 1,
    };
    let encoded = EncodedEntry::new(&entry).unwrap();
    assert_eq!(encoded.table, 2);
    assert_eq!(encoded.key_hash, key_hash(&entry.key));
    assert_eq!(encoded.frame, encode_entry(&entry).unwrap());
}
