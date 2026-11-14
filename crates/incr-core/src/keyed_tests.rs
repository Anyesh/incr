use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::Arc;

use proptest::prelude::*;

use super::*;
use crate::persist_hooks::RepSide;
use crate::{ErasedIncr, Incr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    KeyHash,
    KeyEq,
    RowEq,
    Clone,
}

type Hook = (Call, Box<dyn FnOnce()>);

thread_local! {
    static ARMED: Cell<Option<(Call, usize)>> = const { Cell::new(None) };
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// Panics at the `nth` call of an armed kind, once, and runs a hook
/// installed for the kind at its first call.
fn tick(call: Call) {
    ARMED.with(|armed| {
        if let Some((armed_call, n)) = armed.get() {
            if armed_call == call {
                if n == 1 {
                    armed.set(None);
                    panic!("chosen {call:?} call");
                }
                armed.set(Some((armed_call, n - 1)));
            }
        }
    });
    let hook = HOOK.with(|h| {
        let mut h = h.borrow_mut();
        if h.as_ref().is_some_and(|(c, _)| *c == call) {
            h.take()
        } else {
            None
        }
    });
    if let Some((_, f)) = hook {
        f();
    }
}

/// Equal and hashed by `id` only, so which object a collection keeps is
/// visible through `tag`.
#[derive(Debug)]
struct Key {
    id: u8,
    tag: u8,
}

impl Clone for Key {
    fn clone(&self) -> Self {
        tick(Call::Clone);
        Key {
            id: self.id,
            tag: self.tag,
        }
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        tick(Call::KeyEq);
        self.id == other.id
    }
}

impl Eq for Key {}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        tick(Call::KeyHash);
        self.id.hash(state);
    }
}

/// Equal and hashed by `val` only.
#[derive(Debug)]
struct Row {
    val: u8,
    tag: u8,
}

impl Clone for Row {
    fn clone(&self) -> Self {
        tick(Call::Clone);
        Row {
            val: self.val,
            tag: self.tag,
        }
    }
}

impl PartialEq for Row {
    fn eq(&self, other: &Self) -> bool {
        tick(Call::RowEq);
        self.val == other.val
    }
}

impl Eq for Row {}

impl Hash for Row {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.val.hash(state);
    }
}

fn k(id: u8) -> Key {
    Key { id, tag: 0 }
}

fn kt(id: u8, tag: u8) -> Key {
    Key { id, tag }
}

fn r(val: u8) -> Row {
    Row { val, tag: 0 }
}

fn rt_(val: u8, tag: u8) -> Row {
    Row { val, tag }
}

type Kc<C> = KeyedCollection<Key, Row, C, (Key, Row), u8>;
type Quad = (u8, u8, u8, u8);

fn quad((key, row): &(Key, Row)) -> Quad {
    (key.id, key.tag, row.val, row.tag)
}

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort_unstable();
    v
}

#[derive(Debug, PartialEq, Eq)]
struct Snap {
    items: Vec<Quad>,
    index: Vec<(Quad, u8)>,
    pairs: Vec<Quad>,
    rows: Vec<(u8, u8)>,
    epoch: u64,
    len: usize,
    settled: (usize, usize),
}

fn snap<C: Cells>(rt: &Runtime<C>, kc: &Kc<C>) -> Snap {
    let index = kc.index.table.read().all_pairs();
    Snap {
        items: sorted(kc.items().iter().map(quad).collect()),
        index: sorted(index.iter().map(|(p, rank)| (quad(p), *rank)).collect()),
        pairs: sorted(kc.pairs().values(rt).iter().map(quad).collect()),
        rows: sorted(
            kc.rows()
                .values(rt)
                .iter()
                .map(|r| (r.val, r.tag))
                .collect(),
        ),
        epoch: kc.epoch(),
        len: kc.len(),
        settled: kc.index().settled(),
    }
}

fn ids(keys: &[Key]) -> Vec<u8> {
    sorted(keys.iter().map(|k| k.id).collect())
}

fn counts(inserted: usize, replaced: usize, removed: usize) -> KeyCounts {
    KeyCounts {
        inserted,
        replaced,
        removed,
    }
}

fn panic_text(err: Box<dyn std::any::Any + Send>) -> String {
    match err.downcast::<String>() {
        Ok(s) => *s,
        Err(err) => err
            .downcast::<&str>()
            .map(|s| s.to_string())
            .unwrap_or_default(),
    }
}

fn apply<C: Cells>(
    rt: &Runtime<C>,
    kc: &Kc<C>,
    ops: Vec<PairOp<(Key, Row), u8>>,
) -> KeyedChange<Key> {
    kc.apply_pairs_with(rt, ops, &mut (), None)
}

fn an_equal_row_changes_nothing_downstream<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Kc::<C>::new(&rt);
    kc.upsert(&rt, k(1), rt_(5, 0));
    let count = kc.rows().count(&rt);
    assert_eq!(rt.get(count), 1);
    let stamp = rt.changed_at(count);
    let pairs_stamp = rt.changed_at(kc.pairs().version_node());
    let epoch = kc.epoch();

    let change = kc.upsert(&rt, kt(1, 3), rt_(5, 9));
    assert_eq!(change.counts, KeyCounts::default());
    assert!(change.changed.is_empty());
    assert_eq!(change.epoch, epoch);
    assert_eq!(kc.epoch(), epoch);
    assert_eq!(rt.get(count), 1);
    assert_eq!(rt.changed_at(count), stamp);
    assert_eq!(rt.changed_at(kc.pairs().version_node()), pairs_stamp);
    assert_eq!(kc.row(&k(1)).map(|row| row.tag), Some(0));
}

fn a_held_key_object_is_kept<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Kc::<C>::new(&rt);
    kc.upsert(&rt, kt(1, 7), r(1));
    let change = kc.upsert(&rt, kt(1, 9), r(2));
    assert_eq!(change.counts, counts(0, 1, 0));
    assert_eq!(change.changed[0].tag, 7);
    assert_eq!(kc.keys().iter().map(|k| k.tag).collect::<Vec<_>>(), vec![7]);
    assert_eq!(
        sorted(kc.items().iter().map(quad).collect()),
        vec![(1, 7, 2, 0)]
    );
    assert_eq!(
        kc.pairs().values(&rt).iter().map(quad).collect::<Vec<_>>(),
        vec![(1, 7, 2, 0)]
    );
}

fn reconcile_makes_the_rows_equal_the_pairs<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Kc::<C>::new(&rt);
    for id in 1..=3 {
        kc.upsert(&rt, k(id), r(id * 10));
    }
    let change = kc.reconcile(
        &rt,
        [(k(1), r(10)), (k(2), r(99)), (k(4), r(40)), (k(2), r(21))],
    );
    assert_eq!(change.counts, counts(1, 1, 1));
    assert_eq!(ids(&change.changed), vec![2, 3, 4]);
    assert_eq!(kc.len(), 3);
    for (id, val) in [(1, Some(10)), (2, Some(21)), (3, None), (4, Some(40))] {
        assert_eq!(kc.row(&k(id)).map(|row| row.val), val, "key {id}");
    }
    assert_eq!(
        sorted(kc.rows().values(&rt).iter().map(|row| row.val).collect()),
        vec![10, 21, 40]
    );

    let epoch = kc.epoch();
    let change = kc.reconcile(
        &rt,
        [(k(1), r(10)), (k(2), r(5)), (k(2), r(21)), (k(4), r(40))],
    );
    assert_eq!(change.counts, KeyCounts::default());
    assert!(change.changed.is_empty());
    assert_eq!(kc.epoch(), epoch);

    let change = kc.reconcile(&rt, Vec::new());
    assert_eq!(change.counts, counts(0, 0, 3));
    assert!(kc.is_empty());
    assert!(kc.pairs().values(&rt).is_empty());
}

/// Each write a test runs against a populated collection: key 2 holds two
/// pairs, so a write can drop a pair that is not read.
fn populated<C: Cells>(rt: &Runtime<C>) -> Kc<C> {
    let kc = Kc::<C>::new(rt);
    for id in 1..=3 {
        kc.upsert(rt, k(id), r(id * 10));
    }
    apply(rt, &kc, vec![PairOp::Insert((k(2), r(22)), 4)]);
    kc
}

fn write<C: Cells>(rt: &Runtime<C>, kc: &Kc<C>, which: usize) {
    match which {
        0 => drop(kc.upsert(rt, k(2), r(7))),
        1 => drop(kc.upsert(rt, k(5), r(50))),
        2 => drop(kc.remove(rt, &k(1))),
        3 => drop(kc.reconcile(rt, [(k(1), r(11)), (k(4), r(40)), (k(1), r(12))])),
        _ => drop(apply(
            rt,
            kc,
            vec![
                PairOp::Remove((k(3), r(30)), 0),
                PairOp::Insert((k(3), r(31)), 2),
                PairOp::Insert((k(6), r(60)), 0),
                PairOp::Remove((k(2), r(20)), 0),
            ],
        )),
    }
}

/// The sweep builds one runtime per panic point, at about a second each
/// under Miri, so Miri runs the upsert, reconcile and batch writes against
/// the key equality and clone calls only.
fn panic_sweep() -> (&'static [Call], &'static [usize]) {
    if cfg!(miri) {
        (&[Call::KeyEq, Call::Clone], &[0, 3, 4])
    } else {
        (
            &[Call::KeyHash, Call::KeyEq, Call::RowEq, Call::Clone],
            &[0, 1, 2, 3, 4],
        )
    }
}

fn user_panics_leave_everything_unchanged<C: Cells>() {
    let (calls, writes) = panic_sweep();
    for &call in calls {
        let mut panicked = 0;
        for &which in writes {
            for nth in 1.. {
                let rt = Runtime::<C>::new();
                let kc = populated(&rt);
                let before = snap(&rt, &kc);
                ARMED.with(|a| a.set(Some((call, nth))));
                let result = catch_unwind(AssertUnwindSafe(|| write(&rt, &kc, which)));
                ARMED.with(|a| a.set(None));
                let Err(err) = result else { break };
                panicked += 1;
                assert!(panic_text(err).contains("chosen"));
                assert_eq!(snap(&rt, &kc), before, "{call:?} {nth} in write {which}");
                write(&rt, &kc, which);
                assert!(snap(&rt, &kc).epoch > before.epoch);
            }
        }
        assert!(
            panicked >= writes.len(),
            "{call:?} panicked only {panicked} times"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Reentry {
    RowOfSame,
    LenOfSame,
    UpsertSame,
    UpsertOther,
}

fn runtime_calls_from_key_or_row_eq_are_refused<C: Cells>() {
    for call in [Call::KeyEq, Call::RowEq] {
        for reentry in [
            Reentry::RowOfSame,
            Reentry::LenOfSame,
            Reentry::UpsertSame,
            Reentry::UpsertOther,
        ] {
            let rt = Rc::new(Runtime::<C>::new());
            let kc = populated(&rt);
            let other = populated(&rt);
            let before = (snap(&rt, &kc), snap(&rt, &other));
            let hook: Box<dyn FnOnce()> = {
                let (rt, kc, other) = (Rc::clone(&rt), kc.clone(), other.clone());
                Box::new(move || match reentry {
                    Reentry::RowOfSame => drop(kc.row(&k(1))),
                    Reentry::LenOfSame => drop(kc.len()),
                    Reentry::UpsertSame => drop(kc.upsert(&rt, k(9), r(9))),
                    Reentry::UpsertOther => drop(other.upsert(&rt, k(9), r(9))),
                })
            };
            HOOK.with(|h| *h.borrow_mut() = Some((call, hook)));
            let err = catch_unwind(AssertUnwindSafe(|| kc.upsert(&rt, k(1), r(1))))
                .expect_err("a reentrant call was let through");
            assert!(HOOK.with(|h| h.borrow().is_none()), "{call:?} never ran");
            let text = panic_text(err);
            assert!(
                text.contains("KeyedCollection"),
                "{call:?} {reentry:?}: {text}"
            );
            assert!(text.contains(value_code::CONTRACT_MESSAGE), "{text}");
            assert_eq!(
                (snap(&rt, &kc), snap(&rt, &other)),
                before,
                "{call:?} {reentry:?}"
            );
            kc.upsert(&rt, k(1), r(1));
            assert_eq!(kc.row(&k(1)).map(|row| row.val), Some(1));
        }
    }
}

#[derive(Default)]
struct Reps {
    added: usize,
    removed: usize,
}

impl<T> RepSink<T> for Reps {
    fn on_rep(&mut self, side: RepSide, _: &T) {
        match side {
            RepSide::Added => self.added += 1,
            RepSide::Removed => self.removed += 1,
        }
    }
}

fn a_remove_then_insert_of_one_key_reports_two_changes<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Kc::<C>::new(&rt);
    kc.upsert(&rt, k(1), r(10));
    let mut reps = Reps::default();
    let change = kc.apply_pairs_with(
        &rt,
        [
            PairOp::Remove((k(1), r(10)), 0),
            PairOp::Insert((k(1), r(11)), 0),
        ],
        &mut reps,
        None,
    );
    assert_eq!(ids(&change.changed), vec![1, 1]);
    assert_eq!(change.counts, counts(1, 0, 1));
    assert_eq!((reps.added, reps.removed), (1, 1));
    assert_eq!(kc.row(&k(1)).map(|row| row.val), Some(11));
}

fn a_key_with_two_pairs_reads_the_smallest_rank<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Kc::<C>::new(&rt);
    let change = apply(
        &rt,
        &kc,
        vec![
            PairOp::Insert((k(1), r(10)), 5),
            PairOp::Insert((k(1), r(20)), 2),
        ],
    );
    assert_eq!(change.counts, counts(1, 1, 0));
    assert_eq!(kc.row(&k(1)).map(|row| row.val), Some(20));
    assert_eq!(kc.len(), 1);
    assert_eq!(kc.pairs().values(&rt).len(), 2);
    assert_eq!(kc.rows().values(&rt).len(), 2);

    let change = apply(&rt, &kc, vec![PairOp::Insert((k(1), r(30)), 2)]);
    assert!(change.changed.is_empty());
    assert_eq!(kc.row(&k(1)).map(|row| row.val), Some(20));
    apply(&rt, &kc, vec![PairOp::Remove((k(1), r(20)), 2)]);
    assert_eq!(kc.row(&k(1)).map(|row| row.val), Some(30));

    let change = kc.upsert(&rt, k(1), r(40));
    assert_eq!(change.counts, counts(0, 1, 0));
    assert_eq!(kc.pairs().values(&rt).len(), 1);
}

fn removing_an_absent_pair_is_a_no_op<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Kc::<C>::new(&rt);
    kc.upsert(&rt, k(1), r(10));
    let before = snap(&rt, &kc);
    let mut journal = Vec::new();
    let change = kc.apply_pairs_with(
        &rt,
        [
            PairOp::Remove((k(1), r(11)), 0),
            PairOp::Remove((k(2), r(10)), 0),
        ],
        &mut (),
        Some(&mut journal),
    );
    assert_eq!(change.counts, KeyCounts::default());
    assert!(change.changed.is_empty());
    assert_eq!(change.epoch, before.epoch);
    assert!(journal.is_empty());
    assert_eq!(snap(&rt, &kc), before);
}

fn an_inverted_journal_restores_the_previous_state<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = populated(&rt);
    let before = snap(&rt, &kc);
    let mut journal = Vec::new();
    kc.apply_pairs_with(
        &rt,
        [
            PairOp::Remove((kt(1, 5), r(10)), 0),
            PairOp::Insert((k(1), r(12)), 3),
            PairOp::Insert((k(4), r(40)), 0),
            PairOp::Remove((k(2), r(22)), 0),
            PairOp::Remove((k(9), r(90)), 0),
            PairOp::Insert((k(3), r(30)), 0),
        ],
        &mut (),
        Some(&mut journal),
    );
    assert_eq!(journal.len(), 5);
    assert!(matches!(&journal[0], PairOp::Remove((key, _), 0) if key.tag == 0));
    assert!(matches!(&journal[3], PairOp::Remove((_, row), 4) if row.val == 22));
    assert_ne!(snap(&rt, &kc), before);
    kc.apply_pairs_with(
        &rt,
        journal.into_iter().rev().map(PairOp::inverse),
        &mut (),
        None,
    );
    let after = snap(&rt, &kc);
    assert_eq!(
        Snap {
            epoch: before.epoch,
            ..after
        },
        before
    );
}

fn journaled_writes_are_undone_by_their_inverted_journal<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = populated(&rt);
    let before = snap(&rt, &kc);
    let mut journal = Vec::new();
    kc.upsert_journaled(&rt, kt(1, 7), r(11), Some(&mut journal));
    kc.upsert_journaled(&rt, k(3), r(30), Some(&mut journal));
    kc.remove_journaled(&rt, &k(2), Some(&mut journal));
    kc.reconcile_journaled(&rt, [(k(1), r(12)), (k(5), r(50))], Some(&mut journal));
    assert_eq!(journal.len(), 8);
    assert!(matches!(&journal[1], PairOp::Insert((key, row), 0) if key.tag == 0 && row.val == 11));
    kc.apply_pairs_with(
        &rt,
        journal.into_iter().rev().map(PairOp::inverse),
        &mut (),
        None,
    );
    let after = snap(&rt, &kc);
    assert_eq!(
        Snap {
            epoch: before.epoch,
            ..after
        },
        before
    );
}

fn changed_lists_the_keys_whose_row_differs_and_epochs_rise<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Kc::<C>::new(&rt);
    assert_eq!(kc.epoch(), 1);
    let steps: Vec<(KeyedChange<Key>, Vec<u8>, u64)> = vec![
        (kc.upsert(&rt, k(1), r(1)), vec![1], 2),
        (kc.upsert(&rt, k(1), r(1)), vec![], 2),
        (kc.upsert(&rt, k(2), r(2)), vec![2], 3),
        (kc.remove(&rt, &k(9)), vec![], 3),
        (kc.upsert(&rt, k(2), r(3)), vec![2], 4),
        (kc.remove(&rt, &k(1)), vec![1], 5),
        (
            kc.reconcile(&rt, [(k(2), r(3)), (k(3), r(3)), (k(4), r(4))]),
            vec![3, 4],
            6,
        ),
    ];
    for (i, (change, changed, epoch)) in steps.iter().enumerate() {
        assert_eq!(ids(&change.changed), *changed, "step {i}");
        assert_eq!(change.epoch, *epoch, "step {i}");
        let c = change.counts;
        assert_eq!(
            c.inserted + c.replaced + c.removed,
            changed.len(),
            "step {i}"
        );
    }
}

fn keys_and_items_are_snapshots_and_len_counts_keys<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Kc::<C>::new(&rt);
    kc.upsert(&rt, k(1), r(1));
    kc.upsert(&rt, k(2), r(2));
    let (keys, items) = (kc.keys(), kc.items());
    kc.remove(&rt, &k(1));
    kc.upsert(&rt, k(3), r(3));
    assert_eq!(ids(&keys), vec![1, 2]);
    assert_eq!(
        sorted(items.iter().map(quad).collect()),
        vec![(1, 0, 1, 0), (2, 0, 2, 0)]
    );
    apply(&rt, &kc, vec![PairOp::Insert((k(2), r(9)), 1)]);
    assert_eq!(kc.len(), 2);
    assert_eq!(kc.pairs().values(&rt).len(), 3);
    assert_eq!(ids(&kc.keys()), vec![2, 3]);
    assert!(kc.contains(&k(2)) && !kc.contains(&k(1)));
}

fn settled_stats_are_read_without_the_lock<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Kc::<C>::new(&rt);
    assert_eq!(kc.index().settled(), (0, 0));
    for id in 0..20 {
        kc.upsert(&rt, k(id), r(id));
    }
    apply(&rt, &kc, vec![PairOp::Insert((k(1), r(9)), 1)]);
    let (pairs, bytes) = kc.index().settled();
    assert_eq!(pairs, 21);
    assert!(bytes > 0);
    let _held = kc.index.table.write();
    assert_eq!(kc.index().settled(), (pairs, bytes));
    let stats = kc.index().state_stats();
    assert_eq!((stats.elements, stats.bytes), (pairs, bytes));
}

macro_rules! for_both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(#[test] fn $name() { super::$name::<crate::cells::Local>(); })*
        }
        mod shared {
            $(#[test] fn $name() { super::$name::<crate::cells::Shared>(); })*
        }
    };
    (@rows $($name:ident),* $(,)?) => {
        mod rows_local {
            $(#[test] fn $name() { super::$name::<crate::cells::Local>(); })*
        }
        mod rows_shared {
            $(#[test] fn $name() { super::$name::<crate::cells::Shared>(); })*
        }
    };
}

for_both!(
    an_equal_row_changes_nothing_downstream,
    a_held_key_object_is_kept,
    reconcile_makes_the_rows_equal_the_pairs,
    user_panics_leave_everything_unchanged,
    runtime_calls_from_key_or_row_eq_are_refused,
    a_remove_then_insert_of_one_key_reports_two_changes,
    a_key_with_two_pairs_reads_the_smallest_rank,
    removing_an_absent_pair_is_a_no_op,
    an_inverted_journal_restores_the_previous_state,
    journaled_writes_are_undone_by_their_inverted_journal,
    changed_lists_the_keys_whose_row_differs_and_epochs_rise,
    keys_and_items_are_snapshots_and_len_counts_keys,
    settled_stats_are_read_without_the_lock,
    model_matches,
);

type Plain<C> = KeyedCollection<u8, u8, C, (u8, u8), u8>;
type Model = BTreeMap<u8, Vec<((u8, u8), u8)>>;

#[derive(Clone, Debug)]
enum Step {
    Upsert(u8, u8),
    Remove(u8),
    Reconcile(Vec<(u8, u8)>),
    Apply(Vec<(bool, u8, u8, u8)>),
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        (0..5u8, 0..3u8).prop_map(|(k, v)| Step::Upsert(k, v)),
        (0..5u8).prop_map(Step::Remove),
        prop::collection::vec((0..5u8, 0..3u8), 0..6).prop_map(Step::Reconcile),
        prop::collection::vec((any::<bool>(), 0..5u8, 0..3u8, 0..3u8), 0..6).prop_map(Step::Apply),
    ]
}

fn model_read(model: &Model, key: u8) -> Option<u8> {
    let held = model.get(&key)?;
    first_smallest(held.iter().map(|e| e.1)).map(|i| held[i].0 .1)
}

fn model_upsert(model: &mut Model, key: u8, row: u8) {
    if model_read(model, key) != Some(row) {
        model.insert(key, vec![((key, row), 0)]);
    }
}

fn model_apply(model: &mut Model, ops: &[PairOp<(u8, u8), u8>]) -> Vec<PairOp<(u8, u8), u8>> {
    let mut journal = Vec::new();
    for op in ops {
        match op {
            PairOp::Insert(pair, rank) => {
                model.entry(pair.0).or_default().push((*pair, *rank));
                journal.push(op.clone());
            }
            PairOp::Remove(pair, rank) => {
                let Some(held) = model.get_mut(&pair.0) else {
                    continue;
                };
                let found = held
                    .iter()
                    .position(|e| e == &(*pair, *rank))
                    .or_else(|| held.iter().position(|e| e.0 == *pair));
                if let Some(i) = found {
                    let (pair, rank) = held.remove(i);
                    journal.push(PairOp::Remove(pair, rank));
                }
                if held.is_empty() {
                    model.remove(&pair.0);
                }
            }
        }
    }
    journal
}

fn flat(model: &Model) -> Vec<((u8, u8), u8)> {
    sorted(model.values().flatten().copied().collect())
}

fn assert_holds<C: Cells>(rt: &Runtime<C>, kc: &Plain<C>, model: &Model) {
    assert_eq!(sorted(kc.index.table.read().all_pairs()), flat(model));
    let pairs: Vec<(u8, u8)> = flat(model).iter().map(|e| e.0).collect();
    assert_eq!(sorted(kc.pairs().values(rt)), sorted(pairs.clone()));
    let rows: Vec<u8> = pairs.iter().map(|p| p.1).collect();
    assert_eq!(sorted(kc.rows().values(rt)), sorted(rows));
    for key in 0..5 {
        assert_eq!(kc.row(&key), model_read(model, key), "row of {key}");
    }
    assert_eq!(kc.len(), model.len());
    assert_eq!(sorted(kc.keys()), model.keys().copied().collect::<Vec<_>>());
}

fn changed_reads(before: &Model, after: &Model) -> Vec<u8> {
    (0..5)
        .filter(|&key| model_read(before, key) != model_read(after, key))
        .collect()
}

fn run_steps<C: Cells>(steps: Vec<Step>) {
    let rt = Runtime::<C>::new();
    let kc = Plain::<C>::new(&rt);
    let mut model = Model::new();
    for step in steps {
        let before = model.clone();
        let epoch = kc.epoch();
        let (change, written) = match &step {
            Step::Upsert(key, row) => {
                model_upsert(&mut model, *key, *row);
                (kc.upsert(&rt, *key, *row), model.clone())
            }
            Step::Remove(key) => {
                model.remove(key);
                (kc.remove(&rt, key), model.clone())
            }
            Step::Reconcile(rows) => {
                let target: BTreeMap<u8, u8> = rows.iter().copied().collect();
                model.retain(|key, _| target.contains_key(key));
                for (&key, &row) in &target {
                    model_upsert(&mut model, key, row);
                }
                (kc.reconcile(&rt, rows.iter().copied()), model.clone())
            }
            Step::Apply(ops) => {
                let ops: Vec<PairOp<(u8, u8), u8>> = ops
                    .iter()
                    .map(|&(insert, key, row, rank)| match insert {
                        true => PairOp::Insert((key, row), rank),
                        false => PairOp::Remove((key, row), rank),
                    })
                    .collect();
                let mut journal = Vec::new();
                let change = kc.apply_pairs_with(&rt, ops.clone(), &mut (), Some(&mut journal));
                let expected = model_apply(&mut model, &ops);
                assert_eq!(journal, expected);
                assert_holds(&rt, &kc, &model);
                let written = model.clone();
                let undo: Vec<_> = journal.into_iter().rev().map(PairOp::inverse).collect();
                let mut undone = model.clone();
                model_apply(&mut undone, &undo);
                kc.apply_pairs_with(&rt, undo, &mut (), None);
                assert_holds(&rt, &kc, &undone);
                assert_eq!(flat(&undone), flat(&before));
                model = undone;
                model_apply(&mut model, &ops);
                kc.apply_pairs_with(&rt, ops, &mut (), None);
                (change, written)
            }
        };
        assert_holds(&rt, &kc, &model);
        let differs = changed_reads(&before, &written);
        let c = change.counts;
        assert_eq!(c.inserted + c.replaced + c.removed, change.changed.len());
        match step {
            Step::Apply(_) => {
                for key in &differs {
                    assert!(
                        change.changed.contains(key),
                        "{key} missing from {:?}",
                        change.changed
                    );
                }
            }
            _ => {
                assert_eq!(sorted(change.changed.clone()), differs);
                assert_eq!(change.epoch, epoch + u64::from(!differs.is_empty()));
            }
        }
        if flat(&before) != flat(&written) {
            assert!(change.epoch > epoch);
        }
        assert!(kc.epoch() >= change.epoch);
    }
}

fn model_matches<C: Cells>() {
    let config = ProptestConfig {
        cases: if cfg!(miri) { 6 } else { 64 },
        failure_persistence: None,
        ..ProptestConfig::default()
    };
    proptest!(config, |(steps in prop::collection::vec(step(), 1..12))| {
        run_steps::<C>(steps);
    });
}

thread_local! {
    static TRACKED_DROPS: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Tracked(u32);

impl Drop for Tracked {
    fn drop(&mut self) {
        TRACKED_DROPS.set(TRACKED_DROPS.get() + 1);
    }
}

#[test]
fn apply_pairs_keeping_drops_nothing_until_the_kept_value_drops() {
    let rt = Runtime::<crate::Local>::new();
    let kc = KeyedCollection::<u32, Tracked, crate::Local, (u32, Tracked), u8>::new(&rt);
    kc.apply_pairs_with(
        &rt,
        [
            PairOp::Insert((1, Tracked(1)), 0),
            PairOp::Insert((2, Tracked(2)), 0),
        ],
        &mut (),
        None,
    );
    let mut keep = KeyedSpent::default();
    TRACKED_DROPS.set(0);
    kc.apply_pairs_keeping(
        &rt,
        [
            PairOp::Remove((1, Tracked(1)), 0),
            PairOp::Insert((1, Tracked(3)), 0),
            PairOp::Remove((2, Tracked(2)), 0),
            PairOp::Remove((9, Tracked(9)), 0),
        ],
        &mut (),
        None,
        &mut keep,
    );
    kc.apply_pairs_keeping(
        &rt,
        [PairOp::Remove((9, Tracked(9)), 0)],
        &mut (),
        None,
        &mut keep,
    );
    assert_eq!(TRACKED_DROPS.get(), 0);
    assert!(!kc.contains(&2));
    assert_eq!(kc.row(&1), Some(Tracked(3)));
    TRACKED_DROPS.set(0);
    drop(keep);
    assert!(TRACKED_DROPS.get() >= 4, "{} dropped", TRACKED_DROPS.get());
}

/// Lets a query closure, which must be `Send + Sync`, capture a keyed
/// collection.
struct Captured<T>(T);

// SAFETY: every test runs a `Local` runtime and its collections on the one
// thread that created them; `Shared` collections are `Send + Sync` anyway.
unsafe impl<T> Send for Captured<T> {}
// SAFETY: see `Send`.
unsafe impl<T> Sync for Captured<T> {}

impl<T> Captured<T> {
    fn get(&self) -> &T {
        &self.0
    }
}

type Runs = Arc<std::sync::atomic::AtomicUsize>;

fn runs_of(runs: &Runs) -> usize {
    runs.load(std::sync::atomic::Ordering::SeqCst)
}

/// A query reading `key` through `get_key`, and how often it ran.
fn key_reader<C: Cells>(rt: &Runtime<C>, kc: &Plain<C>, key: u8) -> (Incr<Option<u8>>, Runs) {
    let runs: Runs = Arc::default();
    let counted = Arc::clone(&runs);
    let kc = Captured(kc.clone());
    let q = rt.create_query(move |rt| {
        counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        kc.get().get_key(rt, &key)
    });
    (q, runs)
}

fn a_key_reader_reruns_only_when_its_row_changes<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Plain::<C>::new(&rt);
    kc.upsert(&rt, 1, 10);
    kc.upsert(&rt, 2, 20);
    let (a, runs) = key_reader(&rt, &kc, 1);
    assert_eq!(rt.get(a), Some(10));
    assert_eq!(runs_of(&runs), 1);

    kc.upsert(&rt, 2, 21);
    assert_eq!(rt.get(a), Some(10));
    assert_eq!(runs_of(&runs), 1, "a write to another key");

    kc.upsert(&rt, 1, 10);
    assert_eq!(rt.get(a), Some(10));
    assert_eq!(runs_of(&runs), 1, "an equal row");

    kc.upsert(&rt, 1, 11);
    assert_eq!(rt.get(a), Some(11));
    assert_eq!(runs_of(&runs), 2);

    kc.reconcile(&rt, [(1, 11), (3, 30)]);
    assert_eq!(rt.get(a), Some(11));
    assert_eq!(runs_of(&runs), 2, "a reconcile leaving the key's row");
}

fn an_absent_key_reader_follows_removal_and_re_add<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Plain::<C>::new(&rt);
    let (q, runs) = key_reader(&rt, &kc, 3);
    assert_eq!(rt.get(q), None);
    kc.upsert(&rt, 3, 30);
    assert_eq!(rt.get(q), Some(30));
    kc.remove(&rt, &3);
    assert_eq!(rt.get(q), None);
    kc.upsert(&rt, 3, 31);
    assert_eq!(rt.get(q), Some(31));
    assert_eq!(runs_of(&runs), 4);
    kc.remove(&rt, &4);
    assert_eq!(rt.get(q), Some(31));
    assert_eq!(runs_of(&runs), 4);
}

fn replayed_pairs_and_their_undo_rerun_the_reader<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Plain::<C>::new(&rt);
    kc.upsert(&rt, 1, 10);
    let (q, runs) = key_reader(&rt, &kc, 1);
    assert_eq!(rt.get(q), Some(10));
    let mut journal = Vec::new();
    kc.upsert_journaled(&rt, 1, 12, Some(&mut journal));
    assert_eq!(rt.get(q), Some(12));
    kc.apply_pairs_with(
        &rt,
        journal.into_iter().rev().map(PairOp::inverse),
        &mut (),
        None,
    );
    assert_eq!(rt.get(q), Some(10));
    kc.apply_pairs_with(&rt, [PairOp::Insert((1, 9), 0)], &mut (), None);
    assert_eq!(rt.get(q), Some(10), "a higher-ranked pair is not read");
    assert_eq!(runs_of(&runs), 3);
}

fn get_key_outside_a_query_creates_nothing<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Plain::<C>::new(&rt);
    kc.upsert(&rt, 1, 10);
    assert_eq!(kc.get_key(&rt, &1), Some(10));
    assert_eq!(kc.get_key(&rt, &2), None);
    assert_eq!(kc.version_input(&1), None);
    assert_eq!(kc.version_input(&2), None);
}

/// `combine` runs under `aggregate`'s state lock, so a `get_key` there
/// creates a version input under it.
fn get_key_inside_an_aggregate_combine_creates_its_input<C: Cells>() {
    let rt = Arc::new(Runtime::<C>::new());
    let kc = Plain::<C>::new(&rt);
    kc.upsert(&rt, 1, 10);
    let numbers = IncrCollection::<u8, C>::new(&rt);
    numbers.insert(&rt, 1);
    numbers.insert(&rt, 2);
    let captured = Captured((kc.clone(), Arc::clone(&rt)));
    let total = numbers.aggregate(
        &rt,
        0_u32,
        |n| u32::from(*n),
        move |a, b| {
            let (kc, rt) = captured.get();
            a + b + u32::from(kc.get_key(rt, &1).unwrap_or(0))
        },
    );
    rt.get(total);
    numbers.insert(&rt, 3);
    rt.get(total);
    assert!(kc.version_input(&1).is_some());
}

fn version_inputs_map_back_to_their_keys<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Plain::<C>::new(&rt);
    kc.upsert(&rt, 1, 10);
    let (q, _) = key_reader(&rt, &kc, 1);
    let (absent, _) = key_reader(&rt, &kc, 7);
    rt.get(q);
    rt.get(absent);
    let one = kc.version_input(&1).expect("read in a query");
    let seven = kc.version_input(&7).expect("read in a query");
    assert_eq!(kc.key_of_version(one.erase()), Some(1));
    assert_eq!(kc.key_of_version(seven.erase()), Some(7));
    assert_eq!(kc.key_of_version(kc.pairs().version_node().erase()), None);
    let all: std::collections::HashSet<ErasedIncr> = kc.version_inputs().into_iter().collect();
    assert_eq!(all, [one.erase(), seven.erase()].into_iter().collect());
}

fn a_removed_key_without_readers_is_reclaimed_on_local_only<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kc = Plain::<C>::new(&rt);
    kc.upsert(&rt, 1, 10);
    kc.upsert(&rt, 2, 20);
    let (gone, _) = key_reader(&rt, &kc, 1);
    let (kept, _) = key_reader(&rt, &kc, 2);
    rt.get(gone);
    rt.get(kept);
    let handle = kc.version_input(&1).unwrap();
    rt.try_delete_node(gone).unwrap();
    kc.remove(&rt, &1);
    kc.remove(&rt, &2);
    if C::SINGLE_THREADED {
        assert_eq!(kc.version_input(&1), None);
        assert!(!rt.is_alive(handle));
        assert_eq!(kc.key_of_version(handle.erase()), None);
    } else {
        assert_eq!(kc.version_input(&1), Some(handle));
        assert!(rt.is_alive(handle));
    }
    assert!(kc.version_input(&2).is_some(), "a key with a reader");
    assert_eq!(rt.get(kept), None);
}

fn a_keyed_write_inside_a_compute_is_refused<C: Cells>() {
    let rt = Arc::new(Runtime::<C>::new());
    let kc = Plain::<C>::new(&rt);
    let captured = Captured(kc.clone());
    let q = rt.create_query(move |rt| {
        catch_unwind(AssertUnwindSafe(|| captured.get().upsert(rt, 1, 1)))
            .map_err(panic_text)
            .err()
    });
    let text = rt.get(q).expect("the write panics");
    assert!(text.contains("inside a compute"), "{text}");
    assert_eq!(kc.row(&1), None);

    let other = Runtime::<C>::new();
    let captured = Captured((kc.clone(), Arc::clone(&rt)));
    let q = other.create_query(move |_| {
        let (kc, rt) = captured.get();
        catch_unwind(AssertUnwindSafe(|| kc.upsert(rt, 1, 1)))
            .map_err(panic_text)
            .err()
    });
    let text = other
        .get(q)
        .expect("a write inside another runtime's compute panics");
    assert!(text.contains("inside a compute"), "{text}");
    assert_eq!(kc.row(&1), None);
}

fn one_upsert_among_many_keys_reruns_one_reader<C: Cells>() {
    const KEYS: u32 = 240_000;
    let rt = Runtime::<C>::new();
    let kc = KeyedCollection::<u32, u32, C>::new(&rt);
    kc.reconcile(&rt, (0..KEYS).map(|key| (key, key)));
    let runs: Runs = Arc::default();
    let readers: Vec<Incr<Option<u32>>> = (0..KEYS)
        .step_by(240)
        .map(|key| {
            let counted = Arc::clone(&runs);
            let kc = Captured(kc.clone());
            rt.create_query(move |rt| {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                kc.get().get_key(rt, &key)
            })
        })
        .collect();
    for &q in &readers {
        rt.get(q);
    }
    let before = runs_of(&runs);
    kc.upsert(&rt, 480, 7);
    for &q in &readers {
        rt.get(q);
    }
    assert_eq!(runs_of(&runs) - before, 1);
    assert_eq!(rt.get(readers[2]), Some(7));
}

for_both!(
    @rows
    a_key_reader_reruns_only_when_its_row_changes,
    an_absent_key_reader_follows_removal_and_re_add,
    replayed_pairs_and_their_undo_rerun_the_reader,
    get_key_outside_a_query_creates_nothing,
    get_key_inside_an_aggregate_combine_creates_its_input,
    version_inputs_map_back_to_their_keys,
    a_removed_key_without_readers_is_reclaimed_on_local_only,
    a_keyed_write_inside_a_compute_is_refused,
    one_upsert_among_many_keys_reruns_one_reader,
);
