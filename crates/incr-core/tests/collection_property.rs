//! Property tests for incremental collection operators. Each test
//! generates a random sequence of insert/delete operations on a source
//! collection, runs the incremental pipeline, and compares against a
//! from-scratch batch computation over the same final element set.
//!
//! Both `Local` and `Shared` strategies run the same generator + verifier
//! through separate proptest! cases so failures shrink in the correct
//! type context.

use incr_core::{Cells, IncrCollection, Local, Runtime, Shared};
use proptest::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
enum Op {
    Insert(i64),
    Delete(i64),
}

fn apply_to_baseline(ops: &[Op]) -> HashMap<i64, usize> {
    let mut bag: HashMap<i64, usize> = HashMap::new();
    for op in ops {
        match op {
            Op::Insert(v) => *bag.entry(*v).or_insert(0) += 1,
            Op::Delete(v) => {
                if let Some(count) = bag.get_mut(v) {
                    *count -= 1;
                    if *count == 0 {
                        bag.remove(v);
                    }
                }
            }
        }
    }
    bag
}

fn run_filter_count<C: Cells>(ops: &[Op]) -> u64
where
    Runtime<C>: Default,
{
    let rt: Runtime<C> = Runtime::new();
    let c: IncrCollection<i64, C> = rt.create_collection();
    let evens = c.filter(&rt, |x| x % 2 == 0);
    let n = evens.count(&rt);
    for op in ops {
        match op {
            Op::Insert(v) => c.insert(&rt, *v),
            Op::Delete(v) => {
                c.delete(&rt, v);
            }
        }
    }
    rt.get(n)
}

fn batch_filter_count(ops: &[Op]) -> u64 {
    let bag = apply_to_baseline(ops);
    bag.iter()
        .filter(|(v, _)| *v % 2 == 0)
        .map(|(_, n)| *n as u64)
        .sum()
}

fn run_map_reduce_sum<C: Cells>(ops: &[Op]) -> i64
where
    Runtime<C>: Default,
{
    let rt: Runtime<C> = Runtime::new();
    let c: IncrCollection<i64, C> = rt.create_collection();
    let doubled = c.map(&rt, |x| x * 2);
    let total = doubled.reduce(&rt, |xs| xs.iter().sum::<i64>());
    for op in ops {
        match op {
            Op::Insert(v) => c.insert(&rt, *v),
            Op::Delete(v) => {
                c.delete(&rt, v);
            }
        }
    }
    rt.get(total)
}

fn batch_map_reduce_sum(ops: &[Op]) -> i64 {
    let bag = apply_to_baseline(ops);
    bag.iter().map(|(v, n)| (*v * 2) * (*n as i64)).sum()
}

fn run_sort_then_count<C: Cells>(ops: &[Op]) -> usize
where
    Runtime<C>: Default,
{
    let rt: Runtime<C> = Runtime::new();
    let c: IncrCollection<i64, C> = rt.create_collection();
    let sorted = c.sort_by_key(&rt, |x| *x);
    for op in ops {
        match op {
            Op::Insert(v) => c.insert(&rt, *v),
            Op::Delete(v) => {
                c.delete(&rt, v);
            }
        }
    }
    let _ = rt.get(sorted.version_node());
    sorted.snapshot_len()
}

fn batch_count(ops: &[Op]) -> usize {
    let bag = apply_to_baseline(ops);
    bag.values().sum()
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (-50_i64..50).prop_map(Op::Insert),
        (-50_i64..50).prop_map(Op::Delete),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn local_filter_count_matches_batch(ops in prop::collection::vec(op_strategy(), 0..40)) {
        let incremental = run_filter_count::<Local>(&ops);
        let batch = batch_filter_count(&ops);
        prop_assert_eq!(incremental, batch);
    }

    #[test]
    fn shared_filter_count_matches_batch(ops in prop::collection::vec(op_strategy(), 0..40)) {
        let incremental = run_filter_count::<Shared>(&ops);
        let batch = batch_filter_count(&ops);
        prop_assert_eq!(incremental, batch);
    }

    #[test]
    fn local_map_reduce_matches_batch(ops in prop::collection::vec(op_strategy(), 0..40)) {
        let incremental = run_map_reduce_sum::<Local>(&ops);
        let batch = batch_map_reduce_sum(&ops);
        prop_assert_eq!(incremental, batch);
    }

    #[test]
    fn shared_map_reduce_matches_batch(ops in prop::collection::vec(op_strategy(), 0..40)) {
        let incremental = run_map_reduce_sum::<Shared>(&ops);
        let batch = batch_map_reduce_sum(&ops);
        prop_assert_eq!(incremental, batch);
    }

    #[test]
    fn local_sort_preserves_count(ops in prop::collection::vec(op_strategy(), 0..40)) {
        let len = run_sort_then_count::<Local>(&ops);
        let batch = batch_count(&ops);
        prop_assert_eq!(len, batch);
    }

    #[test]
    fn shared_sort_preserves_count(ops in prop::collection::vec(op_strategy(), 0..40)) {
        let len = run_sort_then_count::<Shared>(&ops);
        let batch = batch_count(&ops);
        prop_assert_eq!(len, batch);
    }
}
