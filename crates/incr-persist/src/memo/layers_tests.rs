use super::*;
use crate::memo::{
    encode_entry, DeltaHeader, EncodedEntry, GenHeader, MemoBackend, MemoryMemo, Tombstone,
};

fn header(seq: u64) -> GenHeader {
    GenHeader {
        build: 1,
        seq,
        tables: Vec::new(),
    }
}

fn entry(key: &[u8], value: u8) -> Entry {
    Entry {
        table: 0,
        key: key.to_vec(),
        value: vec![value; 4],
        deps: Vec::new(),
        last_used: 0,
    }
}

/// An entry frame filed under `under`'s key hash, as a hash collision
/// would file it.
fn filed(entry: &Entry, under: &[u8], tag: u64) -> EncodedEntry {
    EncodedEntry {
        table: entry.table,
        key_hash: key_hash(under),
        frame: encode_entry(entry, tag).unwrap(),
    }
}

fn delta(id: u64) -> DeltaHeader {
    DeltaHeader {
        header: header(id),
        base: GenId(1),
        parent: GenId(id - 1),
    }
}

struct Store {
    memo: MemoryMemo,
}

impl Store {
    fn new(base: &[Entry]) -> Self {
        let memo = MemoryMemo::new();
        let tag = header(1).frame_tag();
        let frames: Vec<_> = base.iter().map(|e| filed(e, &e.key, tag)).collect();
        memo.write_generation(None, GenId(1), &header(1), &frames)
            .unwrap();
        Store { memo }
    }

    fn delta(&self, id: u64, frames: &[EncodedEntry], tombstones: &[Tombstone]) {
        self.memo
            .write_delta(GenId(id - 1), GenId(id), &delta(id), frames, tombstones)
            .unwrap();
    }

    fn layers(&self) -> Vec<Arc<Generation>> {
        self.memo.load_chain(&[]).unwrap().unwrap().into_layers()
    }

    fn probe(&self, key: &[u8]) -> Option<u8> {
        let probe = probe(&self.layers(), 0, key, |id, item| {
            self.memo.read_entry(id, item)
        });
        match probe.unwrap() {
            Probe::Found(entry) => Some(entry.value[0]),
            Probe::Missing => None,
            Probe::Gone => panic!("a layer is gone"),
            Probe::Corrupt => Some(0xcc),
        }
    }
}

#[test]
fn the_newest_layer_holding_a_key_decides() {
    let store = Store::new(&[entry(b"a", 1), entry(b"b", 1), entry(b"c", 1)]);
    let tag = delta(2).frame_tag();
    store.delta(2, &[filed(&entry(b"a", 2), b"a", tag)], &[]);
    let tag = delta(3).frame_tag();
    store.delta(
        3,
        &[filed(&entry(b"b", 3), b"b", tag)],
        &[Tombstone {
            table: 0,
            key_hash: key_hash(b"c"),
        }],
    );
    assert_eq!(store.probe(b"a"), Some(2));
    assert_eq!(store.probe(b"b"), Some(3));
    assert_eq!(store.probe(b"c"), None, "a tombstone hides the base entry");
    assert_eq!(store.probe(b"d"), None);
    let layers = store.layers();
    assert!(may_hold(&layers, 0, key_hash(b"a")));
    assert!(!may_hold(&layers, 0, key_hash(b"c")));
    assert!(!may_hold(&layers, 0, key_hash(b"d")));
}

#[test]
fn a_key_mismatch_falls_through_and_a_corrupt_frame_does_not() {
    let store = Store::new(&[entry(b"a", 1), entry(b"b", 1)]);
    let tag = delta(2).frame_tag();
    store.delta(
        2,
        &[
            filed(&entry(b"other", 2), b"a", tag),
            filed(&entry(b"b", 2), b"b", tag),
        ],
        &[],
    );
    assert_eq!(store.probe(b"a"), Some(1), "a collision in the delta");
    let layers = store.layers();
    let item = layers[1].find(0, key_hash(b"b"))[0];
    store.memo.replace_frame(GenId(2), &item, vec![0; 12]);
    assert_eq!(store.probe(b"b"), Some(0xcc), "not the base's older value");
}

#[test]
fn a_tombstone_in_the_same_layer_as_a_colliding_entry_hides_only_older_layers() {
    let store = Store::new(&[entry(b"a", 1)]);
    let tag = delta(2).frame_tag();
    store.delta(
        2,
        &[filed(&entry(b"twin", 2), b"a", tag)],
        &[Tombstone {
            table: 0,
            key_hash: key_hash(b"a"),
        }],
    );
    assert_eq!(store.probe(b"twin"), None, "filed under another hash");
    assert_eq!(store.probe(b"a"), None);
    let layers = store.layers();
    let run = layers[1].find(0, key_hash(b"a"));
    assert_eq!(run.len(), 2);
    assert!(may_hold(&layers, 0, key_hash(b"a")));
}

#[test]
fn live_totals_replace_each_shadowed_run() {
    let store = Store::new(&[entry(b"a", 1), entry(b"b", 1), entry(b"c", 1)]);
    let base = Live::of_chain(&store.layers());
    let frame = |e: &Entry, tag| filed(e, &e.key, tag);
    let one = frame(&entry(b"a", 2), 0).frame.len() as u64;
    assert_eq!(
        base,
        Live {
            entries: 3,
            bytes: 3 * one
        }
    );
    let tag = delta(2).frame_tag();
    store.delta(
        2,
        &[frame(&entry(b"a", 2), tag), frame(&entry(b"z", 2), tag)],
        &[],
    );
    let tag = delta(3).frame_tag();
    store.delta(
        3,
        &[frame(&entry(b"a", 3), tag)],
        &[Tombstone {
            table: 0,
            key_hash: key_hash(b"b"),
        }],
    );
    let live = Live::of_chain(&store.layers());
    assert_eq!(
        live,
        Live {
            entries: 3,
            bytes: 3 * one
        }
    );
}
