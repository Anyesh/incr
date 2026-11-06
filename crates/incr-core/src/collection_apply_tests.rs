use std::hash::Hasher;

use super::tests::user_code_panics::{arm, disarm, failure, populated, sorted, Bomb, Call};
use super::*;
use crate::cells::{Local, Shared};

fn sorted_values<T: Value + Hash + Eq + Ord, C: Cells>(
    rt: &Runtime<C>,
    c: &IncrCollection<T, C>,
) -> Vec<T> {
    let mut values = c.values(rt);
    values.sort_unstable();
    values
}

/// How many input sets `between` makes, measured by the revisions of a
/// probe input set just before and just after it.
fn sets_during<C: Cells>(rt: &Runtime<C>, between: impl FnOnce()) -> u64 {
    let probe = rt.create_input(0_u64);
    rt.set(probe, 1);
    let first = rt.changed_at(probe);
    rt.set(probe, 2);
    let second = rt.changed_at(probe);
    let step = second - first;
    between();
    rt.set(probe, 3);
    (rt.changed_at(probe) - second) / step - 1
}

fn one_set_per_call<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let version = rt.get(c.version_node());

    let batch = [
        Change::Insert(2),
        Change::Insert(3),
        Change::Delete(1),
        Change::Insert(2),
    ];
    assert_eq!(sets_during(&rt, || drop(c.apply(&rt, batch))), 1);
    assert_eq!(rt.get(c.version_node()), version + 4);
    assert_eq!(sorted_values(&rt, &c), vec![2, 2, 3]);

    assert_eq!(sets_during(&rt, || c.extend(&rt, [4, 5, 6])), 1);
    assert_eq!(sorted_values(&rt, &c), vec![2, 2, 3, 4, 5, 6]);

    let version = rt.get(c.version_node());
    let nothing = [Change::Insert(9), Change::Delete(9), Change::Delete(42)];
    assert_eq!(sets_during(&rt, || drop(c.apply(&rt, nothing))), 0);
    assert_eq!(sets_during(&rt, || c.extend(&rt, [])), 0);
    assert_eq!(rt.get(c.version_node()), version);
}

#[test]
fn local_apply_sets_the_version_once_per_call() {
    one_set_per_call::<Local>();
}

#[test]
fn shared_apply_sets_the_version_once_per_call() {
    one_set_per_call::<Shared>();
}

#[derive(Debug, PartialEq)]
struct Readout {
    values: Vec<i64>,
    count: u64,
    sum: i64,
    groups: Vec<(i64, Vec<i64>)>,
    joined: Vec<(i64, i64)>,
    mapped: Vec<i64>,
}

struct Pipeline<C: Cells> {
    left: IncrCollection<i64, C>,
    right: IncrCollection<i64, C>,
    count: Incr<u64>,
    sum: Incr<i64>,
    groups: GroupedCollection<i64, i64, C>,
    joined: IncrCollection<(i64, i64), C>,
    mapped: IncrCollection<i64, C>,
}

impl<C: Cells> Pipeline<C> {
    fn new(rt: &Runtime<C>) -> Self {
        let left = rt.create_collection::<i64>();
        let right = rt.create_collection::<i64>();
        Self {
            count: left.count(rt),
            sum: left.aggregate(rt, 0, |x| *x, |a, b| a + b),
            groups: left.group_by(rt, |x| x % 3),
            joined: left.join(rt, &right, |x| x % 4, |y| y % 4),
            mapped: left.filter(rt, |x| x % 2 == 0).map(rt, |x| x * 10),
            left,
            right,
        }
    }

    fn read(&self, rt: &Runtime<C>) -> Readout {
        rt.get(self.groups.version_node());
        let mut keys = self.groups.keys();
        keys.sort_unstable();
        let groups = keys
            .into_iter()
            .filter_map(|k| {
                let members = sorted_values(rt, &self.groups.get_group(&k)?);
                (!members.is_empty()).then_some((k, members))
            })
            .collect();
        Readout {
            values: sorted_values(rt, &self.left),
            count: rt.get(self.count),
            sum: rt.get(self.sum),
            groups,
            joined: sorted_values(rt, &self.joined),
            mapped: sorted_values(rt, &self.mapped),
        }
    }
}

fn operators_match_one_by_one_changes<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let batched = Pipeline::new(&rt);
    let single = Pipeline::new(&rt);
    let _ = (batched.read(&rt), single.read(&rt));

    let insert_all =
        |values: &mut dyn Iterator<Item = i64>| values.map(Change::Insert).collect::<Vec<_>>();
    let batches: Vec<Vec<Change<i64>>> = vec![
        insert_all(&mut (0..20).chain([3, 3, 5])),
        vec![
            Change::Delete(3),
            Change::Insert(21),
            Change::Delete(100),
            Change::Insert(5),
            Change::Delete(5),
            Change::Delete(5),
            Change::Insert(7),
            Change::Delete(7),
            Change::Delete(8),
            Change::Delete(8),
            Change::Insert(8),
        ],
        (0..10)
            .map(Change::Delete)
            .chain((30..40).map(Change::Insert))
            .collect(),
        Vec::new(),
        insert_all(&mut (0..30).map(|x| x % 7)),
    ];
    let right: [&[i64]; 5] = [&[0, 1, 2], &[3, 3], &[], &[5, 6, 7, 8], &[1]];

    for (batch, right) in batches.into_iter().zip(right) {
        for change in batch.clone() {
            match change {
                Change::Insert(x) => single.left.insert(&rt, x),
                Change::Delete(x) => {
                    single.left.delete(&rt, &x);
                }
            }
        }
        for &y in right {
            single.right.insert(&rt, y);
        }
        let _ = batched.left.apply(&rt, batch.clone());
        batched.right.extend(&rt, right.iter().copied());
        assert_eq!(batched.read(&rt), single.read(&rt), "after {batch:?}");
    }
}

#[test]
fn local_apply_operators_match_one_by_one_changes() {
    operators_match_one_by_one_changes::<Local>();
}

#[test]
fn shared_apply_operators_match_one_by_one_changes() {
    operators_match_one_by_one_changes::<Shared>();
}

fn model(mut elements: Vec<i64>, batch: &[(bool, i64)]) -> Vec<i64> {
    for &(insert, x) in batch {
        if insert {
            elements.push(x);
        } else if let Some(i) = elements.iter().position(|&e| e == x) {
            elements.remove(i);
        }
    }
    elements.sort_unstable();
    elements
}

/// `batch` over `initial` with the `nth` call of `call` panicking: a panic
/// in `Hash`, `Clone` or `Eq` must leave the log, the version and every
/// reader as they were, and a panic in `Drop` must come only after the
/// whole batch is published.
fn check_batch<C: Cells>(initial: &[i64], batch: &[(bool, i64)], call: Call, nth: usize) -> bool {
    let rt: Runtime<C> = Runtime::new();
    let c = populated(&rt, initial);
    let count = c.count(&rt);
    let all = c.filter(&rt, |_| true);
    let _ = (rt.get(count), all.values(&rt));
    let end = c.log.read().end();
    let version = rt.get(c.version_node());

    let staged: Vec<Change<Bomb>> = batch
        .iter()
        .map(|&(insert, x)| {
            if insert {
                Change::Insert(Bomb(x))
            } else {
                Change::Delete(Bomb(x))
            }
        })
        .collect();
    arm(call, nth);
    let panicked = failure(|| drop(c.apply(&rt, staged))).is_some();
    disarm();

    let context = format!("{batch:?} over {initial:?}, {call:?} call {nth}");
    let expected = if panicked && call != Call::Drop {
        assert_eq!(c.log.read().end(), end, "log end after {context}");
        assert_eq!(rt.get(c.version_node()), version, "version after {context}");
        let mut before = initial.to_vec();
        before.sort_unstable();
        before
    } else {
        model(initial.to_vec(), batch)
    };
    assert_eq!(sorted(c.values(&rt)), expected, "values after {context}");
    assert_eq!(
        rt.get(count),
        expected.len() as u64,
        "count after {context}"
    );
    assert_eq!(sorted(all.values(&rt)), expected, "filter after {context}");

    c.insert(&rt, Bomb(-1));
    assert_eq!(rt.get(count), expected.len() as u64 + 1, "{context}");
    panicked
}

fn batches_are_atomic_under_user_panics<C: Cells>() {
    let initial = [1, 2, 2, 3];
    let batch = [
        (true, 4),
        (true, 2),
        (false, 1),
        (true, 5),
        (true, 4),
        (false, 9),
        (false, 3),
        (true, 3),
    ];
    for call in [Call::Hash, Call::Clone, Call::Eq, Call::Drop] {
        let mut fired = 0;
        for nth in 1..=batch.len() {
            if check_batch::<C>(&initial, &batch, call, nth) {
                fired += 1;
            }
        }
        if call == Call::Hash {
            assert_eq!(fired, batch.len(), "every Hash call of the batch panics");
        } else {
            assert!(fired > 1, "{call:?} panicked only {fired} times");
        }
    }
}

#[test]
fn local_apply_is_atomic_under_user_panics() {
    batches_are_atomic_under_user_panics::<Local>();
}

#[test]
fn shared_apply_is_atomic_under_user_panics() {
    batches_are_atomic_under_user_panics::<Shared>();
}

fn changes_that_cancel_are_a_no_op<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let count = c.count(&rt);
    let _ = rt.get(count);
    let end = c.log.read().end();
    let version = rt.get(c.version_node());

    let batches = [
        vec![Change::Insert(5), Change::Delete(5)],
        vec![Change::Delete(1), Change::Insert(1)],
        vec![Change::Delete(1), Change::Delete(1), Change::Insert(1)],
        vec![Change::Insert(1), Change::Delete(1)],
    ];
    for batch in batches {
        let applied = c.apply(&rt, batch.clone());
        assert_eq!(applied, Vec::new(), "{batch:?}");
        assert_eq!(c.log.read().end(), end, "{batch:?}");
        assert_eq!(rt.get(c.version_node()), version, "{batch:?}");
        assert_eq!(c.values(&rt), vec![1], "{batch:?}");
        assert_eq!(rt.get(count), 1, "{batch:?}");
    }
}

#[test]
fn local_apply_of_changes_that_cancel_is_a_no_op() {
    changes_that_cancel_are_a_no_op::<Local>();
}

#[test]
fn shared_apply_of_changes_that_cancel_is_a_no_op() {
    changes_that_cancel_are_a_no_op::<Shared>();
}

/// Every value hashes alike, so all of them share one home slot and tag
/// and every lookup walks a single probe run.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Collide(u32);

impl Hash for Collide {
    fn hash<H: Hasher>(&self, _: &mut H) {}
}

fn colliding_values_all_land<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let c = rt.create_collection::<Collide>();
    c.extend(&rt, (0..4).map(Collide));
    let mut model: Vec<u32> = (0..4).collect();

    let batch: Vec<Change<Collide>> = (100..160)
        .map(|x| Change::Insert(Collide(x)))
        .chain([
            Change::Delete(Collide(1)),
            Change::Insert(Collide(2)),
            Change::Insert(Collide(100)),
            Change::Delete(Collide(3)),
            Change::Insert(Collide(1000)),
        ])
        .collect();
    let _ = c.apply(&rt, batch);
    model.extend(100..160);
    model.retain(|&x| x != 1 && x != 3);
    model.extend([2, 100, 1000]);
    model.sort_unstable();
    let got: Vec<u32> = sorted_values(&rt, &c).into_iter().map(|v| v.0).collect();
    assert_eq!(got, model);

    let second: Vec<Change<Collide>> = (0..4)
        .chain(150..170)
        .map(|x| Change::Delete(Collide(x)))
        .chain((200..220).map(|x| Change::Insert(Collide(x))))
        .collect();
    let _ = c.apply(&rt, second);
    for x in (0..4).chain(150..170) {
        if let Some(i) = model.iter().position(|&m| m == x) {
            model.remove(i);
        }
    }
    model.extend(200..220);
    model.sort_unstable();
    let got: Vec<u32> = sorted_values(&rt, &c).into_iter().map(|v| v.0).collect();
    assert_eq!(got, model);

    for x in model {
        assert!(c.delete(&rt, &Collide(x)), "{x} is findable");
    }
    assert_eq!(c.snapshot_len(), 0);
}

#[test]
fn local_apply_lands_values_that_share_one_probe_run() {
    colliding_values_all_land::<Local>();
}

#[test]
fn shared_apply_lands_values_that_share_one_probe_run() {
    colliding_values_all_land::<Shared>();
}

/// Equal by `key` only, so which copy the collection keeps is visible
/// through `tag`.
#[derive(Clone, Debug)]
struct Tagged {
    key: u32,
    tag: u32,
}

impl PartialEq for Tagged {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Tagged {}

impl Hash for Tagged {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

fn tagged(key: u32, tag: u32) -> Tagged {
    Tagged { key, tag }
}

fn applied_tags(applied: Vec<Change<Tagged>>) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = applied
        .into_iter()
        .map(|change| match change {
            Change::Insert(v) => (true, v.key, v.tag),
            Change::Delete(v) => (false, v.key, v.tag),
        })
        .collect();
    out.sort_unstable();
    out
}

fn removal_returns_the_stored_value<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let c = rt.create_collection::<Tagged>();
    c.insert(&rt, tagged(1, 7));
    c.insert(&rt, tagged(2, 8));
    c.insert(&rt, tagged(2, 9));

    let applied = c.apply(
        &rt,
        [
            Change::Delete(tagged(1, 99)),
            Change::Delete(tagged(2, 98)),
            Change::Insert(tagged(3, 5)),
            Change::Delete(tagged(4, 97)),
        ],
    );
    assert_eq!(
        applied_tags(applied),
        vec![(false, 1, 7), (false, 2, 8), (true, 3, 5)]
    );
    let mut left: Vec<(u32, u32)> = c.values(&rt).iter().map(|v| (v.key, v.tag)).collect();
    left.sort_unstable();
    assert_eq!(left, vec![(2, 8), (3, 5)]);
}

#[test]
fn local_apply_removal_returns_the_stored_value() {
    removal_returns_the_stored_value::<Local>();
}

#[test]
fn shared_apply_removal_returns_the_stored_value() {
    removal_returns_the_stored_value::<Shared>();
}
