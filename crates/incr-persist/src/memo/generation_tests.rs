use proptest::prelude::*;

use super::*;
use std::sync::Arc;

use crate::memo::{encode_entry, key_hash, DeltaHeader, EncodedEntry, Entry, GenHeader, IndexItem};
use crate::Error;

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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // Hashes drawn from a few values that share their low and high bits
    // pile items into the same runs and the same probe chains.
    #[test]
    fn find_matches_a_scan_of_every_item(
        raw in prop::collection::vec((0_u32..3, 0_u64..6, any::<u64>()), 0..64),
    ) {
        let spread = |h: u64| h | (h << 32) | (h << 60);
        let items: Vec<_> = raw
            .iter()
            .map(|&(table, hash, loc)| item(table, spread(hash), loc % 1000))
            .collect();
        let generation = Generation::new(GenId(1), header(), items.clone());
        for table in 0..4 {
            for hash in 0..7 {
                let mut found = generation.find(table, spread(hash)).to_vec();
                let mut want: Vec<_> = items
                    .iter()
                    .filter(|i| (i.table, i.key_hash) == (table, spread(hash)))
                    .copied()
                    .collect();
                found.sort_by_key(|i| i.loc);
                want.sort_by_key(|i| i.loc);
                prop_assert_eq!(found, want);
            }
        }
    }
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
    let encoded = EncodedEntry::new(&entry, 5).unwrap();
    assert_eq!(encoded.table, 2);
    assert_eq!(encoded.key_hash, key_hash(&entry.key));
    assert_eq!(encoded.frame, encode_entry(&entry, 5).unwrap());
}

#[test]
fn a_generation_decodes_with_its_headers_frame_tag() {
    let generation = Generation::new(GenId(1), header(), Vec::new());
    assert_eq!(generation.frame_tag(), header().frame_tag());
}

fn delta(seq: u64) -> DeltaHeader {
    DeltaHeader {
        header: GenHeader { seq, ..header() },
        base: GenId(1),
        parent: GenId(seq - 1),
    }
}

#[test]
fn a_delta_generation_takes_its_delta_header_tag() {
    let head = delta(4);
    let generation = Generation::new_delta(
        GenId(4),
        head.clone(),
        vec![item(0, 7, 0), IndexItem::tombstone(1, 9)],
    );
    assert_eq!(generation.id(), GenId(4));
    assert_eq!(generation.header(), &head.header);
    assert_eq!(generation.frame_tag(), head.frame_tag());
    assert_eq!(generation.base(), Some(GenId(1)));
    assert_eq!(generation.parent(), Some(GenId(3)));
    assert!(generation.find(1, 9)[0].is_tombstone());
    assert_eq!(generation.entry_bytes(), 10);

    let full = Generation::new(GenId(1), header(), vec![item(0, 7, 0)]);
    assert_eq!(full.base(), None);
    assert_eq!(full.parent(), None);
    assert_eq!(full.frame_tag(), header().frame_tag());
}

#[test]
fn a_chain_is_a_full_generation_then_deltas_each_on_the_last() {
    let base = Arc::new(Generation::new(GenId(1), header(), Vec::new()));
    let second = Arc::new(Generation::new_delta(GenId(2), delta(2), Vec::new()));
    let third = Arc::new(Generation::new_delta(GenId(3), delta(3), Vec::new()));
    let chain = Chain::new(vec![base.clone(), second.clone(), third.clone()]).unwrap();
    assert_eq!(chain.depth(), 2);
    assert_eq!(chain.base().id(), GenId(1));
    assert_eq!(chain.tip().id(), GenId(3));
    let ids: Vec<_> = chain.layers().iter().map(|l| l.id()).collect();
    assert_eq!(ids, [GenId(1), GenId(2), GenId(3)]);

    let single = Chain::from(Generation::new(GenId(5), header(), Vec::new()));
    assert_eq!(single.depth(), 0);
    assert_eq!(single.tip().id(), GenId(5));

    let other_base = Arc::new(Generation::new_delta(
        GenId(2),
        DeltaHeader {
            base: GenId(7),
            ..delta(2)
        },
        Vec::new(),
    ));
    for layers in [
        vec![],
        vec![second.clone()],
        vec![base.clone(), third.clone()],
        vec![base.clone(), base.clone()],
        vec![base.clone(), other_base],
        vec![base, third, second],
    ] {
        assert!(matches!(Chain::new(layers), Err(Error::Corrupt(_))));
    }
}

#[test]
fn an_empty_delta_holds_nothing_and_keeps_its_chain_place() {
    let base = Arc::new(Generation::new(GenId(1), header(), vec![item(0, 7, 0)]));
    let damaged = Generation::empty_delta(GenId(2), GenId(1), GenId(1));
    assert!(damaged.items().is_empty());
    assert_eq!(
        (damaged.base(), damaged.parent()),
        (Some(GenId(1)), Some(GenId(1)))
    );
    assert_eq!(damaged.header().build, 0);
    let next = Arc::new(Generation::new_delta(GenId(3), delta(3), Vec::new()));
    let chain = Chain::new(vec![base, Arc::new(damaged), next]).unwrap();
    assert_eq!(chain.depth(), 2);
}
