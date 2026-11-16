use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::Mutex;

use super::*;
use crate::memo::{decode_entry, encode_entry, fingerprint, MemoryMemo, StoredDep, TableDesc};
use crate::Error;

const BUILD: u128 = 77;

fn desc(name: &str, version: u32) -> TableDesc {
    TableDesc {
        name: name.into(),
        version,
        key_type: "u8".into(),
        value_type: "u8".into(),
    }
}

fn entry(table: u32, key: u8, value: u8, deps: Vec<StoredDep>) -> Entry {
    Entry {
        table,
        key: vec![key],
        value: vec![value],
        deps,
        last_used: 0,
    }
}

fn on(table: u32, key: u8) -> StoredDep {
    StoredDep::Query {
        table,
        key: vec![key],
        fp: 0,
    }
}

fn input(tables: Vec<TableDesc>, fresh: Vec<Entry>) -> CheckpointInput {
    CheckpointInput {
        build: BUILD,
        tables,
        fresh,
        max_bytes: DEFAULT_MAX_BYTES,
    }
}

/// Every entry of the current generation, sorted by `(table, key)`.
fn stored(memo: &MemoryMemo) -> Vec<Entry> {
    let generation = memo.load().unwrap().unwrap();
    let mut out: Vec<_> = generation
        .items()
        .iter()
        .map(|item| {
            let frame = memo.read_entry(generation.id(), item).unwrap().unwrap();
            decode_entry(&frame, generation.frame_tag()).unwrap()
        })
        .collect();
    out.sort_by(|a, b| (a.table, &a.key).cmp(&(b.table, &b.key)));
    out
}

fn ids(entries: &[Entry]) -> Vec<(u32, u8, u8, u64)> {
    entries
        .iter()
        .map(|e| (e.table, e.key[0], e.value[0], e.last_used))
        .collect()
}

#[test]
fn first_checkpoint_writes_fresh_entries_at_seq_1() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let done = write_checkpoint(
        &memo,
        input(
            tables.clone(),
            vec![entry(0, 1, 10, vec![]), entry(0, 2, 20, vec![on(0, 1)])],
        ),
    )
    .unwrap();
    assert!(done.superseded.is_empty());
    assert_eq!(done.generation.id(), GenId(1));
    assert_eq!(done.generation.header().seq, 1);
    assert_eq!(done.generation.header().build, BUILD);
    assert_eq!(done.generation.header().tables, tables);
    assert_eq!(done.stats.written, 2);
    assert_eq!(ids(&stored(&memo)), [(0, 1, 10, 1), (0, 2, 20, 1)]);
}

#[test]
fn an_entry_whose_query_dep_is_not_kept_is_dropped_with_its_dependents() {
    let memo = MemoryMemo::new();
    let fresh = vec![
        entry(0, 5, 0, vec![on(0, 4)]),
        entry(0, 1, 0, vec![]),
        entry(0, 2, 0, vec![on(0, 1)]),
        entry(0, 3, 0, vec![on(0, 2), on(0, 1)]),
        entry(0, 4, 0, vec![on(0, 9)]),
        entry(0, 6, 0, vec![on(1, 1)]),
    ];
    let done = write_checkpoint(&memo, input(vec![desc("a", 1), desc("b", 1)], fresh)).unwrap();
    assert_eq!(done.stats.unkept, 3);
    assert_eq!(done.stats.written, 3);
    let keys: Vec<_> = stored(&memo).iter().map(|e| e.key[0]).collect();
    assert_eq!(keys, [1, 2, 3]);
}

#[test]
fn a_full_checkpoint_reports_its_fresh_entries_and_appended_bytes() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(
        &memo,
        input(
            tables.clone(),
            (1..=4).map(|k| entry(0, k, k, vec![])).collect(),
        ),
    )
    .unwrap();
    let fresh = vec![
        entry(0, 1, 50, vec![]),
        entry(0, 7, 70, vec![on(0, 1)]),
        entry(0, 8, 80, vec![on(0, 9)]),
    ];
    let done = write_checkpoint(&memo, input(tables, fresh)).unwrap();
    let stats = &done.stats;
    assert_eq!((stats.written, stats.merged, stats.unkept), (5, 3, 1));
    assert_eq!(stats.fresh_written, 2);
    assert_eq!(stats.bytes_appended, done.generation.entry_bytes());
    assert!(stats.bytes_appended > 0);
    assert!(!stats.delta);
    assert!(!stats.compacted);
    assert_eq!((stats.tombstones, stats.chain_depth), (0, 0));
}

#[test]
fn merge_carries_unrewritten_entries_with_their_last_used() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(
        &memo,
        input(
            tables.clone(),
            vec![entry(0, 1, 10, vec![]), entry(0, 2, 20, vec![])],
        ),
    )
    .unwrap();
    let done = write_checkpoint(
        &memo,
        input(
            tables,
            vec![entry(0, 2, 21, vec![]), entry(0, 3, 30, vec![])],
        ),
    )
    .unwrap();
    assert_eq!(done.superseded, [GenId(1)]);
    assert_eq!(done.generation.id(), GenId(2));
    assert_eq!(done.generation.header().seq, 2);
    assert_eq!(done.stats.merged, 1);
    assert_eq!(
        ids(&stored(&memo)),
        [(0, 1, 10, 1), (0, 2, 21, 2), (0, 3, 30, 2)]
    );
}

#[test]
fn a_base_entry_is_merged_when_its_fresh_rewrite_was_not_kept() {
    let memo = MemoryMemo::without_deltas();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, input(tables.clone(), vec![entry(0, 1, 10, vec![])])).unwrap();
    let done =
        write_checkpoint(&memo, input(tables, vec![entry(0, 1, 11, vec![on(0, 9)])])).unwrap();
    assert_eq!(done.stats.unkept, 1);
    assert_eq!(done.stats.merged, 1);
    assert_eq!(ids(&stored(&memo)), [(0, 1, 10, 1)]);
}

fn on_fp(table: u32, key: u8, value: u8) -> StoredDep {
    StoredDep::Query {
        table,
        key: vec![key],
        fp: fingerprint(&[value]),
    }
}

#[test]
fn a_query_dep_the_base_holds_at_its_fp_counts_as_kept() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, input(tables.clone(), vec![entry(0, 1, 10, vec![])])).unwrap();
    let fresh = vec![
        entry(0, 2, 20, vec![on_fp(0, 1, 10)]),
        entry(0, 3, 30, vec![on(0, 2)]),
        entry(0, 4, 40, vec![on_fp(0, 1, 11)]),
        entry(0, 5, 50, vec![on(0, 4)]),
    ];
    let done = write_checkpoint(&memo, input(tables, fresh)).unwrap();
    assert_eq!(done.stats.unkept, 2);
    assert_eq!(done.stats.merged, 1);
    assert_eq!(
        ids(&stored(&memo)),
        [(0, 1, 10, 1), (0, 2, 20, 2), (0, 3, 30, 2)]
    );
    assert_eq!(
        done.fates,
        [Fate::Written, Fate::Written, Fate::Unkept, Fate::Unkept]
    );
}

#[test]
fn a_dep_whose_rewrite_was_not_kept_does_not_count_as_held() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, input(tables.clone(), vec![entry(0, 1, 10, vec![])])).unwrap();
    let fresh = vec![
        entry(0, 1, 11, vec![on(0, 9)]),
        entry(0, 2, 20, vec![on_fp(0, 1, 11)]),
    ];
    let done = write_checkpoint(&memo, input(tables, fresh)).unwrap();
    assert_eq!(done.stats.unkept, 2);
    assert_eq!(ids(&stored(&memo)), [(0, 1, 10, 1)]);
}

#[test]
fn another_build_drops_the_whole_base() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, input(tables.clone(), vec![entry(0, 1, 10, vec![])])).unwrap();
    let mut next = input(tables, vec![entry(0, 2, 20, vec![])]);
    next.build = BUILD + 1;
    let done = write_checkpoint(&memo, next).unwrap();
    assert_eq!(done.stats.merged, 0);
    assert_eq!(done.generation.header().build, BUILD + 1);
    assert_eq!(ids(&stored(&memo)), [(0, 2, 20, 2)]);
}

#[test]
fn a_changed_table_is_dropped_and_undeclared_tables_are_carried_over() {
    let memo = MemoryMemo::new();
    let base_tables = vec![desc("a", 1), desc("b", 1), desc("c", 1)];
    write_checkpoint(
        &memo,
        input(
            base_tables,
            vec![
                entry(0, 1, 10, vec![]),
                entry(1, 1, 11, vec![]),
                entry(2, 1, 12, vec![on(1, 1)]),
                entry(2, 2, 13, vec![on(0, 1)]),
            ],
        ),
    )
    .unwrap();
    let done = write_checkpoint(&memo, input(vec![desc("b", 1), desc("a", 2)], vec![])).unwrap();
    assert_eq!(
        done.generation.header().tables,
        [desc("b", 1), desc("a", 2), desc("c", 1)]
    );
    assert_eq!(done.stats.merged, 2);
    let entries = stored(&memo);
    assert_eq!(ids(&entries), [(0, 1, 11, 1), (2, 1, 12, 1)]);
    assert_eq!(entries[1].deps, [on(0, 1)]);
}

#[test]
fn damaged_base_frames_are_skipped_and_counted() {
    let memo = MemoryMemo::without_deltas();
    let tables = vec![desc("a", 1)];
    let first = write_checkpoint(
        &memo,
        input(
            tables.clone(),
            vec![
                entry(0, 1, 10, vec![]),
                entry(0, 2, 20, vec![]),
                entry(0, 3, 30, vec![]),
                entry(0, 4, 40, vec![]),
            ],
        ),
    )
    .unwrap()
    .generation;
    let items = first.items().to_vec();
    let frame = memo.read_entry(first.id(), &items[2]).unwrap().unwrap();
    let whole = decode_entry(&frame, first.frame_tag()).unwrap();
    let other_generation = encode_entry(&whole, first.frame_tag() ^ 1).unwrap();
    memo.replace_frame(first.id(), &items[2], other_generation);
    let mut flipped = memo.read_entry(first.id(), &items[0]).unwrap().unwrap();
    let last = flipped.len() - 1;
    flipped[last] ^= 1;
    memo.replace_frame(first.id(), &items[0], flipped);
    let other_key = encode_entry(&entry(0, 9, 0, vec![]), first.frame_tag()).unwrap();
    memo.replace_frame(first.id(), &items[1], other_key);

    let done = write_checkpoint(&memo, input(tables, vec![])).unwrap();
    assert_eq!(done.stats.corrupt, 3);
    assert_eq!(done.stats.merged, 1);
    assert_eq!(stored(&memo).len(), 1);
}

#[test]
fn a_held_lock_makes_the_checkpoint_busy_and_writes_nothing() {
    let memo = MemoryMemo::new();
    let guard = memo.lock_checkpoint().unwrap();
    let result = write_checkpoint(
        &memo,
        input(vec![desc("a", 1)], vec![entry(0, 1, 1, vec![])]),
    );
    assert!(matches!(result, Err(Error::Busy)));
    drop(guard);
    assert!(memo.load().unwrap().is_none());
}

#[test]
fn a_zero_budget_evicts_everything() {
    let memo = MemoryMemo::new();
    let mut next = input(
        vec![desc("a", 1)],
        vec![entry(0, 1, 1, vec![]), entry(0, 2, 2, vec![])],
    );
    next.max_bytes = 0;
    let done = write_checkpoint(&memo, next).unwrap();
    assert_eq!(done.stats.evicted, 2);
    assert_eq!(done.stats.written, 0);
    assert!(done.generation.items().is_empty());
}

#[test]
fn eviction_takes_the_oldest_first_with_transitive_dependents() {
    let mut a = entry(0, 1, 0, vec![]);
    a.last_used = 1;
    let mut b = entry(0, 2, 0, vec![on(0, 1)]);
    b.last_used = 5;
    let mut c = entry(0, 3, 0, vec![on(0, 2)]);
    c.last_used = 5;
    let mut d = entry(0, 4, 0, vec![]);
    d.last_used = 3;
    let mut e = entry(0, 5, 0, vec![]);
    e.last_used = 4;
    let entries = [a, b, c, d, e];
    let nodes: Vec<Node> = entries.iter().map(|e| Node::of(e, 10)).collect();
    let dependents = node_edges(&nodes);
    let evict = |skip: &[bool], max, target| evict(&nodes, &dependents, skip, max, target);
    let none = [false; 5];
    assert_eq!(evict(&none, 50, 50), [false; 5]);
    assert_eq!(evict(&none, 49, 49), [true, true, true, false, false]);
    assert_eq!(evict(&none, 20, 20), [true, true, true, false, false]);
    assert_eq!(evict(&none, 19, 19), [true, true, true, true, false]);
    assert_eq!(evict(&none, 0, 0), [true; 5]);
    assert_eq!(evict(&none, 49, 19), [true, true, true, true, false]);
    assert_eq!(evict(&none, 50, 0), [false; 5]);
    let d_gone = [false, false, false, true, false];
    assert_eq!(evict(&d_gone, 39, 39), [true, true, true, false, false]);
}

#[test]
fn eviction_without_deltas_keeps_the_newest_within_max_bytes() {
    let memo = MemoryMemo::without_deltas();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, input(tables.clone(), vec![entry(0, 1, 10, vec![])])).unwrap();
    let newer = entry(0, 2, 20, vec![]);
    let mut next = input(tables, vec![newer.clone()]);
    let any_tag = 0;
    next.max_bytes = encode_entry(
        &Entry {
            last_used: 2,
            ..newer
        },
        any_tag,
    )
    .unwrap()
    .len() as u64;
    let done = write_checkpoint(&memo, next).unwrap();
    assert_eq!(done.stats.evicted, 1);
    assert_eq!(ids(&stored(&memo)), [(0, 2, 20, 2)]);
}

#[test]
fn an_entry_too_large_for_a_frame_is_dropped_with_its_dependents() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let base = vec![entry(0, 2, 2, vec![]), entry(0, 4, 40, vec![on(0, 2)])];
    write_checkpoint(&memo, input(tables.clone(), base)).unwrap();
    let fresh = vec![
        entry(0, 1, 10, vec![]),
        entry(0, 2, 20, vec![]),
        entry(0, 3, 30, vec![on(0, 2)]),
    ];
    let too_large = |e: &Entry, tag| {
        if e.key == [2] {
            Err(Error::FrameTooLarge {
                len: u64::from(u32::MAX) + 1,
            })
        } else {
            EncodedEntry::new(e, tag)
        }
    };
    let done = write_with(&memo, input(tables, fresh), too_large).unwrap();
    assert_eq!(done.mode, Mode::TwoPass { remap: false });
    assert_eq!(
        done.fates,
        [Fate::Written, Fate::Oversized, Fate::Oversized]
    );
    assert_eq!(done.stats.fresh_written, 1);
    assert_eq!(done.stats.merged, 1);
    assert_eq!(done.stats.oversized, 3);
    assert_eq!(done.stats.evicted, 0);
    assert_eq!(done.stats.written, 1);
    assert_eq!(ids(&stored(&memo)), [(0, 1, 10, 2)]);
}

#[test]
fn a_base_whose_seq_cannot_advance_is_replaced_without_merging() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let header = GenHeader {
        build: BUILD,
        seq: u64::MAX,
        tables: tables.clone(),
    };
    let old = EncodedEntry::new(&entry(0, 1, 10, vec![]), header.frame_tag()).unwrap();
    memo.write_generation(None, GenId(1), &header, &[old])
        .unwrap();
    let done = write_checkpoint(&memo, input(tables, vec![entry(0, 2, 20, vec![])])).unwrap();
    assert_eq!(done.generation.id(), GenId(2));
    assert_eq!(done.generation.header().seq, 1);
    assert_eq!(done.stats.merged, 0);
    assert_eq!(ids(&stored(&memo)), [(0, 2, 20, 1)]);
}

#[test]
fn a_base_with_the_largest_id_fails_the_checkpoint_and_writes_nothing() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let header = GenHeader {
        build: BUILD,
        seq: 1,
        tables: tables.clone(),
    };
    memo.write_generation(None, GenId(u64::MAX), &header, &[])
        .unwrap();
    let result = write_checkpoint(&memo, input(tables, vec![entry(0, 1, 10, vec![])]));
    assert!(matches!(result, Err(Error::Corrupt(_))));
    let current = memo.load().unwrap().unwrap();
    assert_eq!((current.id(), current.items().len()), (GenId(u64::MAX), 0));
}

fn wide(table: u32, n: u32, value: u8) -> Entry {
    Entry {
        table,
        key: n.to_le_bytes().to_vec(),
        value: vec![value; 16],
        deps: Vec::new(),
        last_used: 0,
    }
}

fn on_wide(table: u32, n: u32) -> StoredDep {
    StoredDep::Query {
        table,
        key: n.to_le_bytes().to_vec(),
        fp: 0,
    }
}

fn wide_input(
    tables: Vec<TableDesc>,
    keys: impl Iterator<Item = u32>,
    value: u8,
) -> CheckpointInput {
    input(tables, keys.map(|n| wide(0, n, value)).collect())
}

/// Every entry a newest-first probe of the current chain finds, sorted by
/// `(table, key)`, tables in the base's catalog numbering.
fn visible(memo: &MemoryMemo) -> Vec<Entry> {
    let chain = memo.load_chain(&[]).unwrap().unwrap();
    let mut hidden: HashSet<(u32, u64)> = HashSet::new();
    let mut seen: HashSet<(u32, Vec<u8>)> = HashSet::new();
    let mut out = Vec::new();
    for layer in chain.layers().iter().rev() {
        let mut tombstones = Vec::new();
        for item in layer.items() {
            if item.is_tombstone() {
                tombstones.push((item.table, item.key_hash));
                continue;
            }
            if hidden.contains(&(item.table, item.key_hash)) {
                continue;
            }
            let frame = memo.read_entry(layer.id(), item).unwrap().unwrap();
            let entry = decode_entry(&frame, layer.frame_tag()).unwrap();
            if seen.insert((entry.table, entry.key.clone())) {
                out.push(entry);
            }
        }
        hidden.extend(tombstones);
    }
    out.sort_by(|a, b| (a.table, &a.key).cmp(&(b.table, &b.key)));
    out
}

fn depth(memo: &MemoryMemo) -> usize {
    memo.load_chain(&[]).unwrap().unwrap().depth()
}

#[test]
fn a_small_dirty_set_writes_a_delta_on_the_base() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let base = write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    assert!(!base.stats.delta);
    let done = write_checkpoint(&memo, wide_input(tables.clone(), 10..14, 2)).unwrap();
    let stats = &done.stats;
    assert!(stats.delta);
    assert!(!stats.compacted);
    assert_eq!(
        (stats.written, stats.fresh_written, stats.merged),
        (4, 4, 0)
    );
    assert_eq!((stats.tombstones, stats.chain_depth), (0, 1));
    assert_eq!(stats.bytes_appended, done.generation.entry_bytes());
    assert_eq!(done.generation.base(), Some(base.generation.id()));
    assert_eq!(done.generation.parent(), Some(base.generation.id()));
    assert_eq!(done.generation.header().seq, 2);
    assert!(done.superseded.is_empty());
    assert_eq!(done.chain.depth(), 1);
    assert_eq!(done.chain.tip().id(), done.generation.id());

    let all = visible(&memo);
    assert_eq!(all.len(), 400);
    for entry in &all {
        let n = u32::from_le_bytes(entry.key[..].try_into().unwrap());
        let (value, last_used) = if (10..14).contains(&n) {
            (2, 2)
        } else {
            (1, 1)
        };
        assert_eq!(
            (entry.value[0], entry.last_used),
            (value, last_used),
            "key {n}"
        );
    }
}

#[test]
fn one_percent_of_500k_entries_writes_a_delta_of_the_dirty_count() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, wide_input(tables.clone(), 0..500_000, 1)).unwrap();
    let dirty = (0..500_000).step_by(100);
    let done = write_checkpoint(&memo, wide_input(tables, dirty, 2)).unwrap();
    assert!(done.stats.delta);
    assert_eq!(done.stats.fresh_written, 5_000);
    assert_eq!(done.generation.items().len(), 5_000);
}

#[test]
fn a_chain_of_eight_deltas_compacts_on_the_ninth_checkpoint() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    let mut ids = vec![GenId(1)];
    for round in 1..=8 {
        let done =
            write_checkpoint(&memo, wide_input(tables.clone(), round..round + 1, 2)).unwrap();
        assert!(done.stats.delta, "round {round}");
        assert_eq!(done.stats.chain_depth, round as usize);
        ids.push(done.generation.id());
    }
    let done = write_checkpoint(&memo, wide_input(tables, 20..21, 3)).unwrap();
    assert!(!done.stats.delta);
    assert!(done.stats.compacted);
    assert_eq!(done.stats.chain_depth, 0);
    assert_eq!(done.stats.written, 400);
    assert_eq!(done.stats.fresh_written, 1);
    assert_eq!(done.superseded, ids);
    assert_eq!(depth(&memo), 0);
    let all = visible(&memo);
    assert_eq!(all.len(), 400);
    assert!(all.iter().filter(|e| e.value[0] == 2).count() == 8);
}

#[test]
fn a_delta_over_a_quarter_of_the_base_compacts() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    let done = write_checkpoint(&memo, wide_input(tables.clone(), 0..90, 2)).unwrap();
    assert!(done.stats.delta);
    let done = write_checkpoint(&memo, wide_input(tables, 0..101, 3)).unwrap();
    assert!(!done.stats.delta);
    assert!(done.stats.compacted);
    assert_eq!(done.superseded, [GenId(1), GenId(2)]);
    assert_eq!(visible(&memo).len(), 400);
}

#[test]
fn a_delta_that_would_exceed_max_bytes_compacts_and_evicts_to_nine_tenths() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let base = write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    let max_bytes = base.generation.entry_bytes();
    let mut next = wide_input(tables.clone(), 0..10, 2);
    next.max_bytes = max_bytes;
    assert!(write_checkpoint(&memo, next).unwrap().stats.delta);
    let mut over = wide_input(tables.clone(), 1000..1010, 3);
    over.max_bytes = max_bytes;
    let done = write_checkpoint(&memo, over).unwrap();
    assert!(!done.stats.delta);
    assert!(done.stats.compacted);
    assert_eq!(done.mode, Mode::TwoPass { remap: false });
    assert!(done.generation.entry_bytes() <= max_bytes - max_bytes / 10);
    let all = visible(&memo);
    assert_eq!(done.stats.evicted, 410 - all.len());
    assert_eq!(all.iter().filter(|e| e.last_used > 1).count(), 20);

    for round in 0..4 {
        let mut small = wide_input(tables.clone(), 2000 + round..2001 + round, 4);
        small.max_bytes = max_bytes;
        let done = write_checkpoint(&memo, small).unwrap();
        assert!(done.stats.delta, "round {round}");
        assert_eq!(done.stats.evicted, 0);
    }
}

/// A table name, key, value, `last_used` and deps with tables by name.
type Named = (String, Vec<u8>, Vec<u8>, u64, Vec<String>);

/// Every entry `memo`'s chain shows, with tables and query-dep tables by
/// name, so stores whose catalogs number tables differently compare.
fn named(memo: &MemoryMemo) -> Vec<Named> {
    let chain = memo.load_chain(&[]).unwrap().unwrap();
    let names: Vec<String> = chain
        .tip()
        .header()
        .tables
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let mut out: Vec<_> = visible(memo)
        .into_iter()
        .map(|e| {
            let deps = e
                .deps
                .into_iter()
                .map(|dep| match dep {
                    StoredDep::Query { table, key, fp } => {
                        format!("{}:{key:?}:{fp}", names[table as usize])
                    }
                    other => format!("{other:?}"),
                })
                .collect();
            (
                names[e.table as usize].clone(),
                e.key,
                e.value,
                e.last_used,
                deps,
            )
        })
        .collect();
    out.sort();
    out
}

/// What `rounds` leave visible when every fresh entry is kept and none is
/// evicted: each key's latest write, at its round's seq, in `named` form.
fn oracle(rounds: &[CheckpointInput]) -> Vec<Named> {
    let mut latest = HashMap::new();
    for (at, round) in rounds.iter().enumerate() {
        let name = |table: u32| round.tables[table as usize].name.clone();
        for entry in &round.fresh {
            let deps = entry
                .deps
                .iter()
                .map(|dep| match dep {
                    StoredDep::Query { table, key, fp } => {
                        format!("{}:{key:?}:{fp}", name(*table))
                    }
                    other => format!("{other:?}"),
                })
                .collect();
            let value = (entry.value.clone(), at as u64 + 1, deps);
            latest.insert((name(entry.table), entry.key.clone()), value);
        }
    }
    let mut out: Vec<Named> = latest
        .into_iter()
        .map(|((table, key), (value, seq, deps))| (table, key, value, seq, deps))
        .collect();
    out.sort();
    out
}

/// Writes `rounds` to a store with deltas and to one without, checking
/// after each that both show what [`oracle`] expects, and returns each
/// round's mode on the store with deltas.
fn against_full(rounds: Vec<CheckpointInput>) -> Vec<Mode> {
    let chained = MemoryMemo::new();
    let full = MemoryMemo::without_deltas();
    let mut modes = Vec::new();
    for at in 0..rounds.len() {
        let round = rounds[at].clone();
        modes.push(write_checkpoint(&chained, round.clone()).unwrap().mode);
        write_checkpoint(&full, round).unwrap();
        let expected = oracle(&rounds[..=at]);
        assert_eq!(named(&chained), expected, "round {at}");
        assert_eq!(named(&full), expected, "round {at}");
    }
    modes
}

#[test]
fn a_compaction_without_eviction_or_remap_retags_and_matches_a_full_write() {
    let tables = vec![desc("a", 1)];
    let mut rounds = vec![wide_input(tables.clone(), 0..400, 1)];
    for round in 1..=9 {
        let mut next = wide_input(tables.clone(), round..round + 3, round as u8 + 1);
        next.fresh
            .push(entry(0, 7, round as u8, vec![on_wide(0, round)]));
        rounds.push(next);
    }
    let modes = against_full(rounds);
    assert_eq!(modes[0], Mode::Copy { remap: false });
    assert!(modes[1..9].iter().all(|&p| p == Mode::Delta));
    assert_eq!(modes[9], Mode::Copy { remap: false });
}

#[test]
fn a_compaction_over_a_reordered_catalog_remaps_and_matches_a_full_write() {
    let (a, b, c) = (desc("a", 1), desc("b", 1), desc("c", 1));
    let mut first = wide_input(vec![a.clone(), b.clone()], 0..300, 1);
    first.fresh.extend((0..100).map(|n| wide(1, n, 1)));
    first.fresh.push(entry(1, 9, 9, vec![on_wide(0, 3)]));
    let delta = wide_input(vec![b.clone(), a.clone()], 0..1, 2);
    let mut grown = wide_input(vec![c, b, a], 0..2, 3);
    grown.fresh.push(wide(2, 5, 3));
    let modes = against_full(vec![first, delta, grown]);
    assert_eq!(modes[1], Mode::Delta);
    assert_eq!(modes[2], Mode::Copy { remap: true });
}

#[test]
fn eviction_on_a_reordered_catalog_remaps_in_the_second_pass() {
    let (a, b) = (desc("a", 1), desc("b", 1));
    let memo = MemoryMemo::new();
    let base = write_checkpoint(&memo, wide_input(vec![a.clone()], 0..400, 1)).unwrap();
    let mut next = wide_input(vec![b, a], 0..200, 2);
    next.max_bytes = base.generation.entry_bytes();
    for entry in &mut next.fresh {
        entry.table = 1;
    }
    next.fresh.extend((0..100).map(|n| wide(0, n, 2)));
    let done = write_checkpoint(&memo, next).unwrap();
    assert_eq!(done.mode, Mode::TwoPass { remap: true });
    let max_bytes = base.generation.entry_bytes();
    assert!(done.generation.entry_bytes() <= max_bytes - max_bytes / 10);
    let all = visible(&memo);
    assert_eq!(all.iter().filter(|e| e.last_used == 2).count(), 300);
    assert_eq!(done.stats.evicted, 500 - all.len());
}

/// A [`MemoryMemo`] that records, during a streamed write, how many
/// frames it had handed out beyond the entries the write had pulled.
#[derive(Default)]
struct Streaming {
    memo: MemoryMemo,
    writing: AtomicBool,
    read: AtomicUsize,
    ahead: AtomicUsize,
    last_loc: Mutex<HashMap<u64, u64>>,
    backwards: AtomicUsize,
}

impl Streaming {
    fn count(&self, frames: usize) {
        if self.writing.load(SeqCst) {
            self.read.fetch_add(frames, SeqCst);
        }
    }
}

impl MemoBackend for Streaming {
    type Guard = <MemoryMemo as MemoBackend>::Guard;

    fn lock_checkpoint(&self) -> Result<Self::Guard> {
        self.memo.lock_checkpoint()
    }

    fn load(&self) -> Result<Option<Generation>> {
        self.memo.load()
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        self.count(1);
        self.memo.read_entry(generation, item)
    }

    fn read_entries(&self, generation: GenId, items: &[IndexItem]) -> Result<Vec<Option<Vec<u8>>>> {
        self.count(items.len());
        if self.writing.load(SeqCst) {
            let mut last = self.last_loc.lock().unwrap();
            for item in items {
                if last
                    .insert(generation.0, item.loc)
                    .is_some_and(|prev| prev > item.loc)
                {
                    self.backwards.fetch_add(1, SeqCst);
                }
            }
        }
        self.memo.read_entries(generation, items)
    }

    fn write_generation(
        &self,
        _: Option<GenId>,
        _: GenId,
        _: &GenHeader,
        _: &[EncodedEntry],
    ) -> Result<Generation> {
        panic!("a checkpoint writes full generations streamed");
    }

    fn write_generation_streamed(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &mut dyn Iterator<Item = Result<EncodedEntry>>,
    ) -> Result<Generation> {
        self.read.store(0, SeqCst);
        self.backwards.store(0, SeqCst);
        self.last_loc.lock().unwrap().clear();
        self.writing.store(true, SeqCst);
        let mut pulled = Vec::new();
        for entry in entries {
            pulled.push(entry?);
            let ahead = self.read.load(SeqCst).saturating_sub(pulled.len());
            self.ahead.fetch_max(ahead, SeqCst);
        }
        self.writing.store(false, SeqCst);
        self.memo.write_generation(base, id, header, &pulled)
    }

    fn supports_delta(&self) -> bool {
        true
    }

    fn load_chain(&self, known: &[Arc<Generation>]) -> Result<Option<Chain>> {
        self.memo.load_chain(known)
    }

    fn write_delta(
        &self,
        tip: GenId,
        id: GenId,
        header: &DeltaHeader,
        entries: &[EncodedEntry],
        tombstones: &[Tombstone],
    ) -> Result<Generation> {
        self.memo.write_delta(tip, id, header, entries, tombstones)
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        self.memo.remove_generation(generation)
    }
}

#[test]
fn a_compaction_reads_the_chain_in_batches_as_the_write_pulls() {
    let tables = vec![desc("a", 1)];
    for evicting in [false, true] {
        let memo = Streaming::default();
        let base = write_checkpoint(&memo, wide_input(tables.clone(), 0..4000, 1)).unwrap();
        if !evicting {
            for round in 0..8 {
                write_checkpoint(&memo, wide_input(tables.clone(), round..round + 1, 2)).unwrap();
            }
        }
        let fresh = 4000..4020;
        let mut over = wide_input(tables.clone(), fresh.clone(), 3);
        if evicting {
            over.max_bytes = base.generation.entry_bytes();
        }
        let done = write_checkpoint(&memo, over).unwrap();
        assert!(!done.stats.delta);
        assert_eq!(done.stats.compacted, !evicting);
        assert_eq!(
            done.mode,
            if evicting {
                Mode::TwoPass { remap: false }
            } else {
                Mode::Copy { remap: false }
            }
        );
        assert!(
            memo.ahead.load(SeqCst) <= STREAM_BATCH + fresh.len(),
            "{} frames read ahead of the write",
            memo.ahead.load(SeqCst)
        );
        assert_eq!(
            memo.backwards.load(SeqCst),
            0,
            "frames read out of location order within a layer"
        );
    }
}

#[test]
fn a_streamed_compaction_reading_through_the_same_memory_memo_finishes() {
    let memo = Arc::new(MemoryMemo::new());
    let tables = vec![desc("a", 1)];
    write_checkpoint(&*memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    let (sent, done) = std::sync::mpsc::channel();
    let writer = Arc::clone(&memo);
    std::thread::spawn(move || {
        let result = write_checkpoint(&*writer, wide_input(tables, 0..200, 2));
        sent.send(result.map(|done| done.mode)).unwrap();
    });
    let mode = done
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the compaction finished")
        .unwrap();
    assert_eq!(mode, Mode::Copy { remap: false });
}

#[test]
fn a_new_live_table_or_another_build_compacts() {
    let memo = MemoryMemo::new();
    write_checkpoint(&memo, wide_input(vec![desc("a", 1)], 0..400, 1)).unwrap();
    let grown = wide_input(vec![desc("a", 1), desc("b", 1)], 0..2, 2);
    let done = write_checkpoint(&memo, grown).unwrap();
    assert!(!done.stats.delta);
    assert_eq!(done.stats.merged, 398);

    let mut rebuilt = wide_input(vec![desc("a", 1), desc("b", 1)], 0..2, 3);
    rebuilt.build = BUILD + 1;
    let done = write_checkpoint(&memo, rebuilt).unwrap();
    assert!(!done.stats.delta);
    assert_eq!((done.stats.written, done.stats.merged), (2, 0));
}

#[test]
fn a_dropped_or_reordered_table_still_gets_deltas_in_the_base_catalog() {
    let memo = MemoryMemo::new();
    let mut first = wide_input(vec![desc("a", 1), desc("b", 1)], 0..400, 1);
    first.fresh.extend((0..400).map(|n| wide(1, n, 1)));
    write_checkpoint(&memo, first).unwrap();

    let dropped = wide_input(vec![desc("b", 1)], 0..3, 2);
    let done = write_checkpoint(&memo, dropped).unwrap();
    assert!(done.stats.delta);
    assert_eq!(
        done.generation.header().tables,
        [desc("a", 1), desc("b", 1)]
    );
    let item = done.generation.items()[0];
    assert_eq!(item.table, 1);

    let mut reordered = input(
        vec![desc("b", 1), desc("a", 1)],
        vec![wide(1, 7, 3), {
            let mut e = wide(0, 8, 3);
            e.deps = vec![StoredDep::Query {
                table: 1,
                key: 7_u32.to_le_bytes().to_vec(),
                fp: 0,
            }];
            e
        }],
    );
    reordered.max_bytes = DEFAULT_MAX_BYTES;
    let done = write_checkpoint(&memo, reordered).unwrap();
    assert!(done.stats.delta);
    assert_eq!(done.stats.fresh_written, 2);
    let all = visible(&memo);
    let find = |table: u32, n: u32| {
        all.iter()
            .find(|e| e.table == table && e.key == n.to_le_bytes())
            .unwrap()
    };
    assert_eq!(find(0, 7).value[0], 3, "live table a is base table 0");
    let reader = find(1, 8);
    assert_eq!(reader.value[0], 3);
    assert!(matches!(
        &reader.deps[..],
        [StoredDep::Query { table: 0, .. }]
    ));
}

#[test]
fn a_delta_keeps_unkept_entries_in_the_base_and_tombstones_oversized_ones() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    let held = |n: u32, value: u8| StoredDep::Query {
        table: 0,
        key: n.to_le_bytes().to_vec(),
        fp: fingerprint(&[value; 16]),
    };
    let mut fresh = vec![
        wide(0, 1, 2),
        wide(0, 2, 2),
        wide(0, 3, 2),
        wide(0, 4, 2),
        wide(0, 5, 2),
    ];
    fresh[1].deps = vec![held(300, 1)];
    fresh[2].deps = vec![held(301, 9)];
    fresh[4].deps = vec![held(4, 2)];
    let too_large = |e: &Entry, tag| {
        if e.key == 4_u32.to_le_bytes() {
            Err(Error::FrameTooLarge {
                len: u64::from(u32::MAX) + 1,
            })
        } else {
            EncodedEntry::new(e, tag)
        }
    };
    let done = write_with(&memo, input(tables, fresh), too_large).unwrap();
    let stats = &done.stats;
    assert!(stats.delta);
    assert_eq!(stats.unkept, 1);
    assert_eq!(stats.oversized, 2);
    assert_eq!((stats.fresh_written, stats.tombstones), (2, 2));
    assert_eq!(
        done.fates,
        [
            Fate::Written,
            Fate::Written,
            Fate::Unkept,
            Fate::Oversized,
            Fate::Oversized,
        ]
    );
    assert_eq!(
        done.generation
            .items()
            .iter()
            .filter(|i| i.is_tombstone())
            .count(),
        2
    );
    let all = visible(&memo);
    assert_eq!(all.len(), 398);
    let value = |n: u32| {
        all.iter()
            .find(|e| e.key == n.to_le_bytes())
            .map(|e| e.value[0])
    };
    assert_eq!((value(1), value(2), value(3)), (Some(2), Some(2), Some(1)));
    assert_eq!((value(4), value(5), value(6)), (None, None, Some(1)));
}

#[test]
fn compacting_a_chain_merges_its_visible_entries() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    write_checkpoint(&memo, wide_input(tables.clone(), 0..5, 2)).unwrap();
    let too_large = |e: &Entry, tag| {
        if e.key == 7_u32.to_le_bytes() {
            Err(Error::FrameTooLarge {
                len: u64::from(u32::MAX) + 1,
            })
        } else {
            EncodedEntry::new(e, tag)
        }
    };
    write_with(&memo, wide_input(tables.clone(), 3..8, 3), too_large).unwrap();
    assert_eq!(depth(&memo), 2);
    let before = visible(&memo);
    assert_eq!(before.len(), 399);

    let rebuilt = wide_input(tables, 500..601, 4);
    let done = write_checkpoint(&memo, rebuilt).unwrap();
    assert!(done.stats.compacted);
    assert_eq!(done.stats.merged, 399);
    let after: Vec<_> = visible(&memo)
        .into_iter()
        .filter(|e| e.value[0] != 4)
        .collect();
    assert_eq!(after, before);
}

#[test]
fn a_backend_without_deltas_keeps_writing_full_generations() {
    let memo = MemoryMemo::without_deltas();
    assert!(!memo.supports_delta());
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    let done = write_checkpoint(&memo, wide_input(tables, 0..4, 2)).unwrap();
    assert!(!done.stats.delta);
    assert!(!done.stats.compacted);
    assert_eq!((done.stats.written, done.stats.merged), (400, 396));
    assert_eq!(done.superseded, [GenId(1)]);
    assert_eq!(depth(&memo), 0);
}

#[test]
fn a_delta_with_nothing_to_write_publishes_no_layer() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let base = write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    let mut unkept = wide(0, 5, 2);
    unkept.deps = vec![StoredDep::Query {
        table: 0,
        key: 900_u32.to_le_bytes().to_vec(),
        fp: 0,
    }];
    let done = write_checkpoint(&memo, input(tables, vec![unkept])).unwrap();
    assert_eq!((done.stats.unkept, done.stats.written), (1, 0));
    assert_eq!(done.stats.bytes_appended, 0);
    assert_eq!(done.generation.id(), base.generation.id());
    assert_eq!(depth(&memo), 0);
    assert!(done.superseded.is_empty());
}

#[test]
fn the_next_seq_follows_the_newest_seq_in_the_chain() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let base = write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    write_checkpoint(&memo, wide_input(tables.clone(), 0..2, 2)).unwrap();
    let damaged = DeltaHeader {
        header: GenHeader {
            build: 0,
            seq: 0,
            tables: Vec::new(),
        },
        base: base.generation.id(),
        parent: GenId(2),
    };
    memo.write_delta(GenId(2), GenId(3), &damaged, &[], &[])
        .unwrap();
    let done = write_checkpoint(&memo, wide_input(tables, 2..3, 3)).unwrap();
    assert!(done.stats.delta);
    assert_eq!(done.generation.header().seq, 3);
}

#[test]
fn compacting_past_a_corrupt_delta_frame_does_not_restore_the_older_version() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    let delta = write_checkpoint(&memo, wide_input(tables.clone(), 7..8, 2)).unwrap();
    let item = delta.generation.items()[0];
    memo.replace_frame(delta.generation.id(), &item, vec![0; 12]);
    let done = write_checkpoint(&memo, wide_input(tables, 500..601, 3)).unwrap();
    assert!(done.stats.compacted);
    assert_eq!(done.stats.corrupt, 1);
    let all = visible(&memo);
    assert!(all.iter().all(|e| e.key != 7_u32.to_le_bytes()));
    assert_eq!(all.len(), 399 + 101);
}

#[test]
fn compacting_past_a_dropped_newer_version_does_not_restore_the_older_one() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1), desc("b", 1)];
    let mut first = wide_input(tables.clone(), 0..400, 1);
    first.fresh.push(wide(1, 0, 1));
    write_checkpoint(&memo, first).unwrap();
    let mut reader = wide(0, 9, 2);
    reader.deps = vec![StoredDep::Query {
        table: 1,
        key: 0_u32.to_le_bytes().to_vec(),
        fp: fingerprint(&[1; 16]),
    }];
    let done = write_checkpoint(&memo, input(tables, vec![reader])).unwrap();
    assert!(done.stats.delta);

    let bumped = wide_input(vec![desc("a", 1), desc("b", 2)], 500..501, 3);
    let done = write_checkpoint(&memo, bumped).unwrap();
    assert!(done.stats.compacted);
    let all = visible(&memo);
    assert!(
        all.iter().all(|e| e.key != 9_u32.to_le_bytes()),
        "the base's version of a(9) came back"
    );
}

#[test]
fn an_oversized_entry_no_layer_holds_writes_no_tombstone() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let base = write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
    let too_large = |e: &Entry, tag| {
        if e.key == 900_u32.to_le_bytes() {
            Err(Error::FrameTooLarge {
                len: u64::from(u32::MAX) + 1,
            })
        } else {
            EncodedEntry::new(e, tag)
        }
    };
    let done = write_with(&memo, wide_input(tables, 900..901, 2), too_large).unwrap();
    assert_eq!((done.stats.oversized, done.stats.tombstones), (1, 0));
    assert_eq!(done.generation.id(), base.generation.id());
    assert_eq!(done.fates, [Fate::Oversized]);
}

/// A [`MemoryMemo`] that counts the frames it hands out before a full
/// write starts pulling (`probed`, with the `read_entries` calls among
/// them) and in all, and that answers `None` for the item `dropping`
/// names. Without `batches` it reads one frame per call, as a backend
/// with only `read_entry` does.
struct Counting {
    memo: MemoryMemo,
    batches: bool,
    writing: AtomicBool,
    reads: AtomicUsize,
    probed: AtomicUsize,
    probe_calls: AtomicUsize,
    single_probes: AtomicUsize,
    dropping: std::sync::Mutex<Option<(u32, u64)>>,
}

impl Counting {
    fn new(memo: MemoryMemo, batches: bool) -> Self {
        Counting {
            memo,
            batches,
            writing: AtomicBool::new(false),
            reads: AtomicUsize::new(0),
            probed: AtomicUsize::new(0),
            probe_calls: AtomicUsize::new(0),
            single_probes: AtomicUsize::new(0),
            dropping: std::sync::Mutex::new(None),
        }
    }

    fn reset(&self) {
        for counter in [
            &self.reads,
            &self.probed,
            &self.probe_calls,
            &self.single_probes,
        ] {
            counter.store(0, SeqCst);
        }
    }

    fn read(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        self.reads.fetch_add(1, SeqCst);
        if !self.writing.load(SeqCst) {
            self.probed.fetch_add(1, SeqCst);
        }
        if *self.dropping.lock().unwrap() == Some((item.table, item.key_hash)) {
            return Ok(None);
        }
        self.memo.read_entry(generation, item)
    }
}

impl MemoBackend for Counting {
    type Guard = <MemoryMemo as MemoBackend>::Guard;

    fn lock_checkpoint(&self) -> Result<Self::Guard> {
        self.memo.lock_checkpoint()
    }

    fn load(&self) -> Result<Option<Generation>> {
        self.memo.load()
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        if !self.writing.load(SeqCst) {
            self.single_probes.fetch_add(1, SeqCst);
        }
        self.read(generation, item)
    }

    fn read_entries(&self, generation: GenId, items: &[IndexItem]) -> Result<Vec<Option<Vec<u8>>>> {
        if !self.batches {
            return items
                .iter()
                .map(|item| self.read_entry(generation, item))
                .collect();
        }
        if !self.writing.load(SeqCst) {
            self.probe_calls.fetch_add(1, SeqCst);
        }
        items
            .iter()
            .map(|item| self.read(generation, item))
            .collect()
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        self.memo.write_generation(base, id, header, entries)
    }

    fn write_generation_streamed(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &mut dyn Iterator<Item = Result<EncodedEntry>>,
    ) -> Result<Generation> {
        self.writing.store(true, SeqCst);
        let pulled = entries.collect::<Result<Vec<_>>>();
        self.writing.store(false, SeqCst);
        self.memo.write_generation(base, id, header, &pulled?)
    }

    fn supports_delta(&self) -> bool {
        self.memo.supports_delta()
    }

    fn load_chain(&self, known: &[Arc<Generation>]) -> Result<Option<Chain>> {
        self.memo.load_chain(known)
    }

    fn write_delta(
        &self,
        tip: GenId,
        id: GenId,
        header: &DeltaHeader,
        entries: &[EncodedEntry],
        tombstones: &[Tombstone],
    ) -> Result<Generation> {
        self.memo.write_delta(tip, id, header, entries, tombstones)
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        self.memo.remove_generation(generation)
    }
}

fn on_wide_fp(table: u32, n: u32, value: u8) -> StoredDep {
    StoredDep::Query {
        table,
        key: n.to_le_bytes().to_vec(),
        fp: fingerprint(&[value; 16]),
    }
}

#[test]
fn a_backend_without_deltas_at_its_cap_reads_each_base_entry_once_per_checkpoint() {
    let tables = vec![desc("a", 1)];
    let memo = Counting::new(MemoryMemo::without_deltas(), false);
    let reference = MemoryMemo::without_deltas();
    let first = wide_input(tables.clone(), 0..400, 1);
    let base = write_checkpoint(&memo, first.clone()).unwrap();
    write_checkpoint(&reference, first).unwrap();
    let max_bytes = base.generation.entry_bytes();
    let mut chain = base.chain;
    for round in 0..6_u32 {
        let keys = 1000 + 10 * round..1010 + 10 * round;
        let mut next = wide_input(tables.clone(), keys.clone(), 2 + round as u8);
        let mut reader = wide(0, 2000 + round, 9);
        reader.deps = vec![
            on_wide_fp(0, 399 - round, 1),
            on_wide_fp(0, keys.start, 2 + round as u8),
        ];
        next.fresh.push(reader);
        next.max_bytes = max_bytes;
        let base_entries = chain.base().entry_count() as usize;
        memo.reset();
        let done = write_known(&memo, next.clone(), chain.layers()).unwrap();
        let reads = memo.reads.load(SeqCst);
        assert!(
            reads <= base_entries,
            "round {round}: {reads} reads for {base_entries} base entries"
        );
        assert!(matches!(done.mode, Mode::TwoPass { .. }), "round {round}");
        assert!(done.stats.evicted > 0, "round {round}");
        assert!(done.generation.entry_bytes() <= max_bytes, "round {round}");
        assert!(
            done.fates.iter().all(|&f| f == Fate::Written),
            "round {round}"
        );
        write_checkpoint(&reference, next).unwrap();
        assert_eq!(visible(&memo.memo), visible(&reference), "round {round}");
        chain = done.chain;
    }
}

#[test]
fn dep_probes_are_one_batch_without_fresh_deps_and_a_rejected_delta_reuses_them() {
    let tables = vec![desc("a", 1)];
    for rejected in [false, true] {
        let memo = Counting::new(MemoryMemo::new(), true);
        write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
        let deps_each = if rejected { 10 } else { 2 };
        let mut next = wide_input(tables.clone(), 1000..1001, 2);
        for i in 0..30_u32 {
            let mut reader = wide(0, 1100 + i, 3);
            reader.deps = (0..deps_each)
                .map(|j| on_wide_fp(0, (i + j) % 40, 1))
                .chain([on_wide_fp(0, 1000, 2)])
                .collect();
            next.fresh.push(reader);
        }
        memo.reset();
        let done = write_checkpoint(&memo, next).unwrap();
        assert_eq!(done.stats.delta, !rejected);
        assert!(done.fates.iter().all(|&f| f == Fate::Written));
        let distinct = if rejected { 39 } else { 31 };
        assert_eq!(memo.single_probes.load(SeqCst), 0, "rejected {rejected}");
        assert_eq!(memo.probe_calls.load(SeqCst), 1, "rejected {rejected}");
        assert_eq!(memo.probed.load(SeqCst), distinct, "rejected {rejected}");
    }
}

#[test]
fn a_dep_held_at_its_fp_keeps_its_reader_when_its_fresh_rewrite_is_not_kept() {
    for memo in [MemoryMemo::new(), MemoryMemo::without_deltas()] {
        let tables = vec![desc("a", 1)];
        write_checkpoint(&memo, input(tables.clone(), vec![entry(0, 1, 10, vec![])])).unwrap();
        let fresh = vec![
            entry(0, 1, 11, vec![on(0, 9)]),
            entry(0, 2, 20, vec![on_fp(0, 1, 10)]),
        ];
        let done = write_checkpoint(&memo, input(tables, fresh)).unwrap();
        assert_eq!(done.fates, [Fate::Unkept, Fate::Written]);
        assert_eq!(ids(&visible(&memo)), [(0, 1, 10, 1), (0, 2, 20, 2)]);
    }
}

#[test]
fn a_frame_gone_mid_stream_is_skipped_and_the_rest_carried() {
    let tables = vec![desc("a", 1)];
    for evicting in [false, true] {
        let memo = Counting::new(MemoryMemo::without_deltas(), true);
        let base = write_checkpoint(&memo, wide_input(tables.clone(), 0..400, 1)).unwrap();
        let gone = base.generation.items()[300];
        *memo.dropping.lock().unwrap() = Some((gone.table, gone.key_hash));
        let mut next = wide_input(tables.clone(), 1000..1010, 2);
        if evicting {
            next.max_bytes = base.generation.entry_bytes();
        }
        memo.reset();
        let done = write_known(&memo, next, base.chain.layers()).unwrap();
        assert!(memo.reads.load(SeqCst) > 0);
        assert_eq!(done.stats.corrupt, 1, "evicting {evicting}");
        *memo.dropping.lock().unwrap() = None;
        let all = visible(&memo.memo);
        assert!(all.iter().all(|e| key_hash(&e.key) != gone.key_hash));
        let kept = done.stats.written + done.stats.evicted + 1;
        assert_eq!(kept, 410, "evicting {evicting}");
        assert_eq!(all.len(), done.stats.written, "evicting {evicting}");
    }
}
