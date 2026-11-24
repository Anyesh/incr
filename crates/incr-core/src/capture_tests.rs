use std::hash::{Hash, Hasher};
use std::time::Duration;

use proptest::prelude::*;

use crate::collection::Change;
use crate::keyed::{KeyedCollection, PairOp, UNIDENTIFYING_RANK};
use crate::{IncrCollection, Local, Runtime};

/// Equal by `key` only, so which copy a table keeps shows in `id`. Keys
/// from [`CLUMP`] on all hash alike, so they share one probe run.
#[derive(Clone, Debug)]
pub(crate) struct V {
    pub(crate) key: u8,
    pub(crate) id: u32,
}

pub(crate) const CLUMP: u8 = 64;

impl PartialEq for V {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for V {}

impl Hash for V {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.min(CLUMP).hash(state);
    }
}

pub(crate) fn v(key: u8, id: u32) -> V {
    V { key, id }
}

type Plain = IncrCollection<V, Local>;
type Keyed = KeyedCollection<V, u8, Local, (V, u8), u128>;
type PlainRow = ((u8, u32), usize);
type KeyedRow = ((u8, u32), u8, u128);

fn plain_row((value, count): (V, usize)) -> PlainRow {
    ((value.key, value.id), count)
}

fn keyed_row(((key, row), rank): ((V, u8), u128)) -> KeyedRow {
    ((key.key, key.id), row, rank)
}

fn sorted<T: Ord>(mut rows: Vec<T>) -> Vec<T> {
    rows.sort();
    rows
}

/// A capture in progress: the table as it stood at begin, and what the
/// capture has handed over so far.
struct Taken<T> {
    begin: Vec<T>,
    got: Vec<T>,
}

impl<T: Ord + std::fmt::Debug> Taken<T> {
    fn new(begin: Vec<T>) -> Self {
        Self {
            begin: sorted(begin),
            got: Vec::new(),
        }
    }

    fn check(self) {
        assert_eq!(
            sorted(self.got),
            self.begin,
            "the capture differs from the table at begin"
        );
    }
}

/// Adds a step's rows to the capture in progress, checking it once done.
fn step<T: Ord + std::fmt::Debug, R>(
    taken: &mut Option<Taken<T>>,
    step: crate::CaptureStep<R>,
    row: impl Fn(R) -> T,
) {
    let Some(t) = taken.as_mut() else {
        assert!(step.done && step.rows.is_empty());
        return;
    };
    t.got.extend(step.rows.into_iter().map(row));
    if step.done {
        taken.take().expect("a capture in progress").check();
    }
}

#[derive(Clone, Debug)]
enum PlainStep {
    Insert(u8, u32),
    Delete(u8),
    Replace(Option<u8>, u8, u32),
    Batch(Vec<(bool, u8, u32)>),
    /// Inserts this many keys no other step uses, then deletes them, which
    /// grows the table and leaves tombstones for a same-size rehash.
    Churn(u8),
    Begin,
    Scan(usize),
}

fn plain_step() -> impl Strategy<Value = PlainStep> {
    let key = 0..12u8;
    prop_oneof![
        4 => (key.clone(), any::<u32>()).prop_map(|(k, id)| PlainStep::Insert(k, id)),
        2 => key.clone().prop_map(PlainStep::Delete),
        2 => (proptest::option::of(key.clone()), key.clone(), any::<u32>())
            .prop_map(|(old, k, id)| PlainStep::Replace(old, k, id)),
        2 => prop::collection::vec((any::<bool>(), key, any::<u32>()), 0..8).prop_map(PlainStep::Batch),
        1 => (1..40u8).prop_map(PlainStep::Churn),
        2 => Just(PlainStep::Begin),
        3 => (1..6usize).prop_map(PlainStep::Scan),
    ]
}

fn plain_elements(col: &Plain) -> Vec<PlainRow> {
    col.log
        .read()
        .elements
        .iter()
        .map(|(value, count)| plain_row((value.clone(), count)))
        .collect()
}

fn run_plain(steps: Vec<PlainStep>) {
    let rt = Runtime::<Local>::new();
    let col: Plain = rt.create_collection();
    let mut taken: Option<Taken<PlainRow>> = None;
    for s in steps {
        match s {
            PlainStep::Insert(k, id) => col.insert(&rt, v(k, id)),
            PlainStep::Delete(k) => {
                col.delete(&rt, &v(k, 0));
            }
            PlainStep::Replace(old, k, id) => {
                col.replace(&rt, old.map(|o| v(o, 0)).as_ref(), v(k, id));
            }
            PlainStep::Batch(changes) => {
                col.apply(
                    &rt,
                    changes.into_iter().map(|(insert, k, id)| match insert {
                        true => Change::Insert(v(k, id)),
                        false => Change::Delete(v(k, id)),
                    }),
                );
            }
            PlainStep::Churn(n) => {
                for k in 0..n {
                    col.insert(&rt, v(20 + k, 0));
                }
                for k in 0..n {
                    col.delete(&rt, &v(20 + k, 0));
                }
            }
            PlainStep::Begin => {
                if taken.is_none() {
                    col.begin_capture();
                    taken = Some(Taken::new(plain_elements(&col)));
                }
            }
            PlainStep::Scan(unit) => {
                step(
                    &mut taken,
                    col.capture_step_by(Duration::ZERO, unit),
                    plain_row,
                );
            }
        }
    }
    while taken.is_some() {
        step(
            &mut taken,
            col.capture_step_by(Duration::ZERO, 3),
            plain_row,
        );
    }
    assert!(!col.capture_installed());
}

#[derive(Clone, Debug)]
enum KeyedStep {
    Upsert(u8, u32, u8),
    Remove(u8),
    Reconcile(Vec<(u8, u32, u8)>),
    Apply(Vec<(bool, u8, u8, u8)>),
    /// As [`PlainStep::Churn`].
    Churn(u8),
    Begin,
    Scan(usize),
}

fn keyed_step() -> impl Strategy<Value = KeyedStep> {
    let key = 0..12u8;
    let row = 0..3u8;
    prop_oneof![
        4 => (key.clone(), any::<u32>(), row.clone()).prop_map(|(k, id, r)| KeyedStep::Upsert(k, id, r)),
        2 => key.clone().prop_map(KeyedStep::Remove),
        1 => prop::collection::vec((key.clone(), any::<u32>(), row.clone()), 0..8)
            .prop_map(KeyedStep::Reconcile),
        3 => prop::collection::vec((any::<bool>(), key, row, 0..3u8), 0..8).prop_map(KeyedStep::Apply),
        1 => (1..40u8).prop_map(KeyedStep::Churn),
        2 => Just(KeyedStep::Begin),
        3 => (1..6usize).prop_map(KeyedStep::Scan),
    ]
}

/// Rank `level` of a pair: 0 identifies nothing, and a nonzero rank names
/// one pair only.
fn rank_of(key: u8, row: u8, level: u8) -> u128 {
    match level {
        0 => UNIDENTIFYING_RANK,
        _ => u128::from(level) * 1024 + u128::from(key) * 4 + u128::from(row) + 1,
    }
}

fn run_keyed(steps: Vec<KeyedStep>) {
    let rt = Runtime::<Local>::new();
    let kc = Keyed::new(&rt);
    let mut taken: Option<Taken<KeyedRow>> = None;
    for s in steps {
        match s {
            KeyedStep::Upsert(k, id, r) => {
                kc.upsert(&rt, v(k, id), r);
            }
            KeyedStep::Remove(k) => {
                kc.remove(&rt, &v(k, 0));
            }
            KeyedStep::Reconcile(rows) => {
                let mut seen = std::collections::HashSet::new();
                let rows: Vec<(V, u8)> = rows
                    .into_iter()
                    .filter(|(k, _, _)| seen.insert(*k))
                    .map(|(k, id, r)| (v(k, id), r))
                    .collect();
                kc.reconcile(&rt, rows);
            }
            KeyedStep::Apply(ops) => {
                let ops: Vec<PairOp<(V, u8), u128>> = ops
                    .into_iter()
                    .map(|(insert, k, r, level)| {
                        let pair = (v(k, u32::from(level)), r);
                        let rank = rank_of(k, r, level);
                        match insert {
                            true => PairOp::Insert(pair, rank),
                            false => PairOp::Remove(pair, rank),
                        }
                    })
                    .collect();
                kc.apply_pairs_with(&rt, ops, &mut (), None);
            }
            KeyedStep::Churn(n) => {
                for k in 0..n {
                    kc.upsert(&rt, v(20 + k, 0), 0);
                }
                for k in 0..n {
                    kc.remove(&rt, &v(20 + k, 0));
                }
            }
            KeyedStep::Begin => {
                if taken.is_none() {
                    kc.begin_capture();
                    taken = Some(Taken::new(
                        kc.ranked_pairs().into_iter().map(keyed_row).collect(),
                    ));
                }
            }
            KeyedStep::Scan(unit) => {
                step(
                    &mut taken,
                    kc.capture_step_by(Duration::ZERO, unit),
                    keyed_row,
                );
            }
        }
    }
    while taken.is_some() {
        step(&mut taken, kc.capture_step_by(Duration::ZERO, 3), keyed_row);
    }
    assert!(!kc.capture_installed());
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: if cfg!(miri) { 4 } else { 256 },
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

#[test]
fn plain_capture_equals_the_elements_at_begin() {
    let len = if cfg!(miri) { 1..16 } else { 1..60 };
    proptest!(config(), |(steps in prop::collection::vec(plain_step(), len.clone()))| {
        run_plain(steps);
    });
}

#[test]
fn keyed_capture_equals_the_pairs_at_begin() {
    let len = if cfg!(miri) { 1..16 } else { 1..60 };
    proptest!(config(), |(steps in prop::collection::vec(keyed_step(), len.clone()))| {
        run_keyed(steps);
    });
}

/// Writes to every element mid-capture across a growing rehash and then a
/// same-size one, with the scan stopped part-way through each table.
#[test]
fn plain_capture_survives_both_rehashes_mid_scan() {
    let rt = Runtime::<Local>::new();
    let col: Plain = rt.create_collection();
    for k in 0..20 {
        col.insert(&rt, v(k, u32::from(k)));
    }
    col.begin_capture();
    let mut taken = Some(Taken::new(plain_elements(&col)));
    step(
        &mut taken,
        col.capture_step_by(Duration::ZERO, 5),
        plain_row,
    );
    let bytes = |col: &Plain| col.log.read().elements.heap_bytes();
    let rehashes = |col: &Plain| col.log.read().elements.rehashes();
    let small = bytes(&col);
    col.apply(
        &rt,
        (0..5)
            .map(|k| Change::Delete(v(k, 0)))
            .chain((5..10).map(|k| Change::Insert(v(k, 0))))
            .chain((241..246).map(|k| Change::Insert(v(k, 0)))),
    );
    let regrown = bytes(&col);
    assert!(
        regrown > small,
        "the batch that changes keys grows the table"
    );
    let added = clump_tombstones(
        |k| col.insert(&rt, v(k, 0)),
        |k| {
            col.delete(&rt, &v(k, 0));
        },
    );
    let grown = bytes(&col);
    assert!(grown > regrown, "the clump grows the table");
    step(
        &mut taken,
        col.capture_step_by(Duration::ZERO, 7),
        plain_row,
    );
    let before = rehashes(&col);
    col.apply(
        &rt,
        (10..15)
            .map(|k| Change::Delete(v(k, 0)))
            .chain((15..20).map(|k| Change::Insert(v(k, 0))))
            .chain(added.into_iter().map(|k| Change::Insert(v(k, 0)))),
    );
    assert_eq!(rehashes(&col), before + 1, "the batch rehashes");
    assert_eq!(bytes(&col), grown, "at the same size");
    while taken.is_some() {
        step(
            &mut taken,
            col.capture_step_by(Duration::ZERO, 4),
            plain_row,
        );
    }
    assert!(!col.capture_installed());
}

/// Inserts 150 clumped keys into a table that holds 20, growing it to 256
/// slots, then deletes them. They share one probe run, so a delete leaves
/// a tombstone unless it empties the run's tail, and the tombstones leave
/// fewer than 70 inserts before the table must rehash. Returns 70 keys no
/// write used: inserting them in one write rehashes, at the same size,
/// since 90 keys fit in half the 192-key limit.
pub(crate) fn clump_tombstones(mut insert: impl FnMut(u8), mut delete: impl FnMut(u8)) -> Vec<u8> {
    for k in CLUMP..CLUMP + 150 {
        insert(k);
    }
    for k in CLUMP..CLUMP + 150 {
        delete(k);
    }
    (20..CLUMP).chain(CLUMP + 150..CLUMP + 176).collect()
}

#[test]
fn ending_a_capture_removes_it_and_its_rows() {
    let rt = Runtime::<Local>::new();
    let col: Plain = rt.create_collection();
    let kc = Keyed::new(&rt);
    for k in 0..10 {
        col.insert(&rt, v(k, 0));
        kc.upsert(&rt, v(k, 0), 1);
    }
    col.begin_capture();
    kc.begin_capture();
    col.delete(&rt, &v(3, 0));
    kc.remove(&rt, &v(3, 0));
    col.end_capture();
    kc.end_capture();
    assert!(!col.capture_installed());
    assert!(!kc.capture_installed());
    let empty = col.capture_step(Duration::ZERO);
    assert!(empty.done && empty.rows.is_empty());
    let empty = kc.capture_step(Duration::ZERO);
    assert!(empty.done && empty.rows.is_empty());
    col.begin_capture();
    kc.begin_capture();
    let rows: Vec<_> = col.capture_step(Duration::MAX).rows;
    assert_eq!(rows.len(), 9);
    assert_eq!(kc.capture_step(Duration::MAX).rows.len(), 9);
}

#[test]
#[should_panic(expected = "needs a Local runtime")]
fn a_shared_collection_refuses_a_capture() {
    let rt = Runtime::<crate::Shared>::new();
    rt.create_collection::<V>().begin_capture();
}
