use incr_compute::Runtime;
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

fn verify_group_by_incremental_matches_batch(ops: Vec<Op>) {
    let rt = Runtime::new();
    let col = rt.create_collection::<i64>();
    // group by sign: negative -> -1, zero -> 0, positive -> 1
    let grouped = col.group_by(&rt, |x: &i64| x.signum());

    for op in &ops {
        match op {
            Op::Insert(v) => col.insert(&rt, *v),
            Op::Delete(v) => col.delete(&rt, v),
        }
    }

    // Force the version node to stabilize
    let _ = rt.get(grouped.version_node());

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

    for &key in &[-1_i64, 0, 1] {
        let incr_group_elements: std::collections::HashSet<i64> = grouped
            .get_group(&key)
            .map(|g| g.elements())
            .unwrap_or_default();

        let batch_group: std::collections::HashSet<i64> = batch_set
            .iter()
            .filter(|x| x.signum() == key)
            .copied()
            .collect();

        assert_eq!(
            incr_group_elements, batch_group,
            "group_by mismatch for key={}: incr={:?}, batch={:?}",
            key, incr_group_elements, batch_group,
        );
    }
}

fn verify_join_incremental_matches_batch(left_ops: Vec<Op>, right_ops: Vec<Op>) {
    let rt = Runtime::new();
    let left = rt.create_collection::<i64>();
    let right = rt.create_collection::<i64>();
    // Join on absolute value: pairs (l, r) where abs(l) == abs(r)
    let joined = left.join(&rt, &right, |x: &i64| x.abs(), |x: &i64| x.abs());
    let count = joined.count(&rt);

    for op in &left_ops {
        match op {
            Op::Insert(v) => left.insert(&rt, *v),
            Op::Delete(v) => left.delete(&rt, v),
        }
    }
    for op in &right_ops {
        match op {
            Op::Insert(v) => right.insert(&rt, *v),
            Op::Delete(v) => right.delete(&rt, v),
        }
    }

    let _ = rt.get(count);
    let incr_pairs: std::collections::HashSet<(i64, i64)> = joined.elements();

    // Batch oracle
    let mut left_set = std::collections::HashSet::new();
    for op in &left_ops {
        match op {
            Op::Insert(v) => {
                left_set.insert(*v);
            }
            Op::Delete(v) => {
                left_set.remove(v);
            }
        }
    }
    let mut right_set = std::collections::HashSet::new();
    for op in &right_ops {
        match op {
            Op::Insert(v) => {
                right_set.insert(*v);
            }
            Op::Delete(v) => {
                right_set.remove(v);
            }
        }
    }

    // The join output is a multiset (repeated pairs can appear), but we compare
    // as a set since both sides contain distinct values from the HashSet oracle.
    let mut batch_pairs: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
    for &l in &left_set {
        for &r in &right_set {
            if l.abs() == r.abs() {
                batch_pairs.insert((l, r));
            }
        }
    }

    assert_eq!(
        incr_pairs, batch_pairs,
        "join mismatch: incr={:?}, batch={:?}",
        incr_pairs, batch_pairs,
    );
}

fn verify_window_incremental_matches_batch(ops: Vec<Op>, window_size: usize) {
    if window_size == 0 {
        return;
    }
    let rt = Runtime::new();
    let col = rt.create_collection::<i64>();
    let sorted = col.sort_by_key(&rt, |x: &i64| *x);
    let windows = sorted.window(&rt, window_size);
    let count = windows.count(&rt);

    for op in &ops {
        match op {
            Op::Insert(v) => col.insert(&rt, *v),
            Op::Delete(v) => col.delete(&rt, v),
        }
    }

    let _ = rt.get(count);
    let incr_windows: std::collections::HashSet<Vec<i64>> = windows.elements();

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

    let batch_windows: std::collections::HashSet<Vec<i64>> = if batch_sorted.len() >= window_size {
        batch_sorted
            .windows(window_size)
            .map(|w| w.to_vec())
            .collect()
    } else {
        std::collections::HashSet::new()
    };

    assert_eq!(
        incr_windows, batch_windows,
        "window (size={}) mismatch: incr={:?}, batch={:?}",
        window_size, incr_windows, batch_windows,
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn group_by_incremental_matches_batch(
        ops in prop::collection::vec(op_strategy(), 1..50),
    ) {
        verify_group_by_incremental_matches_batch(ops);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn join_incremental_matches_batch(
        left_ops in prop::collection::vec(op_strategy(), 1..30),
        right_ops in prop::collection::vec(op_strategy(), 1..30),
    ) {
        verify_join_incremental_matches_batch(left_ops, right_ops);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn window_incremental_matches_batch(
        ops in prop::collection::vec(op_strategy(), 1..50),
        window_size in 1_usize..6,
    ) {
        verify_window_incremental_matches_batch(ops, window_size);
    }
}

fn verify_pairwise_incremental_matches_batch(ops: Vec<Op>) {
    let rt = Runtime::new();
    let col = rt.create_collection::<i64>();
    let sorted = col.sort_by_key(&rt, |x: &i64| *x);
    let pairs = sorted.pairwise(&rt);
    let pair_count = pairs.count(&rt);

    for op in &ops {
        match op {
            Op::Insert(v) => col.insert(&rt, *v),
            Op::Delete(v) => col.delete(&rt, v),
        }
    }

    let _ = rt.get(pair_count); // forces stabilization of the full chain
    let incr_pairs = pairs.elements();

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
    let batch_pairs: std::collections::HashSet<(i64, i64)> =
        batch_sorted.windows(2).map(|w| (w[0], w[1])).collect();

    assert_eq!(
        incr_pairs, batch_pairs,
        "Pairwise mismatch: incr={:?}, batch={:?}",
        incr_pairs, batch_pairs
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn pairwise_incremental_matches_batch(
        ops in prop::collection::vec(op_strategy(), 1..50),
    ) {
        verify_pairwise_incremental_matches_batch(ops);
    }
}
