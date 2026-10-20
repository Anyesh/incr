use super::*;
use crate::memo::{decode_entry, encode_entry, MemoryMemo, StoredDep, TableDesc};
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
            decode_entry(&memo.read_entry(generation.id(), item).unwrap().unwrap()).unwrap()
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
    assert_eq!(done.replaced, None);
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
    assert_eq!(done.replaced, Some(GenId(1)));
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
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, input(tables.clone(), vec![entry(0, 1, 10, vec![])])).unwrap();
    let done =
        write_checkpoint(&memo, input(tables, vec![entry(0, 1, 11, vec![on(0, 9)])])).unwrap();
    assert_eq!(done.stats.unkept, 1);
    assert_eq!(done.stats.merged, 1);
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
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    let first = write_checkpoint(
        &memo,
        input(
            tables.clone(),
            vec![
                entry(0, 1, 10, vec![]),
                entry(0, 2, 20, vec![]),
                entry(0, 3, 30, vec![]),
            ],
        ),
    )
    .unwrap()
    .generation;
    let items = first.items().to_vec();
    let mut flipped = memo.read_entry(first.id(), &items[0]).unwrap().unwrap();
    let last = flipped.len() - 1;
    flipped[last] ^= 1;
    memo.replace_frame(first.id(), &items[0], flipped);
    let other_key = encode_entry(&entry(0, 9, 0, vec![])).unwrap();
    memo.replace_frame(first.id(), &items[1], other_key);

    let done = write_checkpoint(&memo, input(tables, vec![])).unwrap();
    assert_eq!(done.stats.corrupt, 2);
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
    let sizes = [10, 10, 10, 10, 10];
    assert_eq!(evict(&entries, &sizes, 50), [false; 5]);
    assert_eq!(
        evict(&entries, &sizes, 49),
        [true, true, true, false, false]
    );
    assert_eq!(
        evict(&entries, &sizes, 20),
        [true, true, true, false, false]
    );
    assert_eq!(evict(&entries, &sizes, 19), [true, true, true, true, false]);
    assert_eq!(evict(&entries, &sizes, 0), [true; 5]);
}

#[test]
fn eviction_through_the_driver_keeps_the_newest_within_budget() {
    let memo = MemoryMemo::new();
    let tables = vec![desc("a", 1)];
    write_checkpoint(&memo, input(tables.clone(), vec![entry(0, 1, 10, vec![])])).unwrap();
    let newer = entry(0, 2, 20, vec![]);
    let mut next = input(tables, vec![newer.clone()]);
    next.max_bytes = encode_entry(&Entry {
        last_used: 2,
        ..newer
    })
    .unwrap()
    .len() as u64;
    let done = write_checkpoint(&memo, next).unwrap();
    assert_eq!(done.stats.evicted, 1);
    assert_eq!(ids(&stored(&memo)), [(0, 2, 20, 2)]);
}
