use incr_core::Runtime;
use proptest::prelude::*;

#[derive(Clone, Debug)]
enum Op {
    Insert(i64),
    Delete(i64),
}

fn verify_collection_incremental_matches_batch(ops: Vec<Op>) {
    let rt = Runtime::new();
    let col = rt.create_collection::<i64>();
    let evens = col.filter(&rt, |x| x % 2 == 0);
    let doubled = evens.map(&rt, |x| x * 2);
    let count = doubled.count(&rt);

    for op in &ops {
        match op {
            Op::Insert(v) => col.insert(&rt, *v),
            Op::Delete(v) => col.delete(&rt, v),
        }
    }

    let incr_count = rt.get(count);
    let incr_elements: std::collections::HashSet<i64> = doubled.elements();

    let mut batch_set = std::collections::HashSet::new();
    for op in &ops {
        match op {
            Op::Insert(v) => {
                batch_set.insert(*v);
            }
            Op::Delete(v) => {
                batch_set.remove(v);
            }
        }
    }
    let batch_elements: std::collections::HashSet<i64> = batch_set
        .iter()
        .filter(|x| *x % 2 == 0)
        .map(|x| x * 2)
        .collect();

    assert_eq!(
        incr_count,
        batch_elements.len(),
        "Count mismatch: incr={}, batch={}",
        incr_count,
        batch_elements.len()
    );
    assert_eq!(incr_elements, batch_elements, "Elements mismatch");
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (-100_i64..100).prop_map(Op::Insert),
        (-100_i64..100).prop_map(Op::Delete),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn collection_incremental_matches_batch(
        ops in prop::collection::vec(op_strategy(), 1..50),
    ) {
        verify_collection_incremental_matches_batch(ops);
    }
}

#[test]
fn collection_property_specific_insert_delete_cycle() {
    verify_collection_incremental_matches_batch(vec![
        Op::Insert(2),
        Op::Insert(4),
        Op::Delete(2),
        Op::Insert(6),
        Op::Insert(3),
        Op::Delete(4),
    ]);
}

fn verify_reduce_incremental_matches_batch(ops: Vec<Op>) {
    let rt = Runtime::new();
    let col = rt.create_collection::<i64>();
    let sum = col.reduce(&rt, |elements| -> i64 { elements.iter().sum() });
    let max = col.reduce(&rt, |elements| -> Option<i64> {
        elements.iter().copied().max()
    });

    for op in &ops {
        match op {
            Op::Insert(v) => col.insert(&rt, *v),
            Op::Delete(v) => col.delete(&rt, v),
        }
    }

    let incr_sum = rt.get(sum);
    let incr_max = rt.get(max);

    // Batch oracle
    let mut batch_set = std::collections::HashSet::new();
    for op in &ops {
        match op {
            Op::Insert(v) => {
                batch_set.insert(*v);
            }
            Op::Delete(v) => {
                batch_set.remove(v);
            }
        }
    }
    let batch_sum: i64 = batch_set.iter().sum();
    let batch_max: Option<i64> = batch_set.iter().copied().max();

    assert_eq!(
        incr_sum, batch_sum,
        "Sum mismatch: incr={}, batch={}",
        incr_sum, batch_sum
    );
    assert_eq!(
        incr_max, batch_max,
        "Max mismatch: incr={:?}, batch={:?}",
        incr_max, batch_max
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn reduce_incremental_matches_batch(
        ops in prop::collection::vec(op_strategy(), 1..50),
    ) {
        verify_reduce_incremental_matches_batch(ops);
    }
}

fn verify_sort_incremental_matches_batch(ops: Vec<Op>) {
    let rt = Runtime::new();
    let col = rt.create_collection::<i64>();
    let sorted = col.sort_by_key(&rt, |x: &i64| *x);

    for op in &ops {
        match op {
            Op::Insert(v) => col.insert(&rt, *v),
            Op::Delete(v) => col.delete(&rt, v),
        }
    }

    let _ = rt.get(sorted.version_node());
    let incr_sorted = sorted.entries();

    // Batch oracle
    let mut batch_set = std::collections::HashSet::new();
    for op in &ops {
        match op {
            Op::Insert(v) => {
                batch_set.insert(*v);
            }
            Op::Delete(v) => {
                batch_set.remove(v);
            }
        }
    }
    let mut batch_sorted: Vec<i64> = batch_set.into_iter().collect();
    batch_sorted.sort();

    assert_eq!(
        incr_sorted, batch_sorted,
        "Sort mismatch: incr={:?}, batch={:?}",
        incr_sorted, batch_sorted
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn sort_incremental_matches_batch(
        ops in prop::collection::vec(op_strategy(), 1..50),
    ) {
        verify_sort_incremental_matches_batch(ops);
    }
}
