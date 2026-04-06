use proptest::prelude::*;
use std::collections::HashSet;

use super::runtime::Runtime;

fn oracle_elements(ops: &[(bool, i64)]) -> HashSet<i64> {
    let mut set = HashSet::new();
    for &(is_insert, val) in ops {
        if is_insert {
            set.insert(val);
        } else {
            set.remove(&val);
        }
    }
    set
}

proptest! {
    #[test]
    fn collection_matches_oracle(
        ops in proptest::collection::vec((proptest::bool::ANY, -100i64..100), 0..50)
    ) {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();

        for &(is_insert, val) in &ops {
            if is_insert {
                col.insert(&rt, val);
            } else {
                col.delete(&rt, &val);
            }
        }

        let expected = oracle_elements(&ops);
        let actual = col.elements();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn filter_matches_oracle(
        ops in proptest::collection::vec((proptest::bool::ANY, -100i64..100), 0..50)
    ) {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let evens = col.filter(&rt, |x| x % 2 == 0);

        for &(is_insert, val) in &ops {
            if is_insert {
                col.insert(&rt, val);
            } else {
                col.delete(&rt, &val);
            }
        }

        let _ = rt.get(evens.version_node());
        let expected: HashSet<i64> = oracle_elements(&ops)
            .into_iter()
            .filter(|x| x % 2 == 0)
            .collect();
        let actual = evens.elements();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn sort_matches_oracle(
        ops in proptest::collection::vec((proptest::bool::ANY, -100i64..100), 0..50)
    ) {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);

        for &(is_insert, val) in &ops {
            if is_insert {
                col.insert(&rt, val);
            } else {
                col.delete(&rt, &val);
            }
        }

        let _ = rt.get(sorted.version_node());
        let mut expected: Vec<i64> = oracle_elements(&ops).into_iter().collect();
        expected.sort();
        let actual = sorted.entries();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn pairwise_matches_oracle(
        ops in proptest::collection::vec((proptest::bool::ANY, -100i64..100), 0..30)
    ) {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sorted = col.sort_by_key(&rt, |x: &i64| *x);
        let pairs = sorted.pairwise(&rt);

        for &(is_insert, val) in &ops {
            if is_insert {
                col.insert(&rt, val);
            } else {
                col.delete(&rt, &val);
            }
        }

        let _ = rt.get(pairs.version_node());
        let mut sorted_vals: Vec<i64> = oracle_elements(&ops).into_iter().collect();
        sorted_vals.sort();
        let expected: HashSet<(i64, i64)> = sorted_vals
            .windows(2)
            .map(|w| (w[0], w[1]))
            .collect();
        let actual = pairs.elements();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn count_matches_oracle(
        ops in proptest::collection::vec((proptest::bool::ANY, -100i64..100), 0..50)
    ) {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let count = col.count(&rt);

        for &(is_insert, val) in &ops {
            if is_insert {
                col.insert(&rt, val);
            } else {
                col.delete(&rt, &val);
            }
        }

        let expected = oracle_elements(&ops).len() as u64;
        let actual = rt.get(count);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn reduce_sum_matches_oracle(
        ops in proptest::collection::vec((proptest::bool::ANY, -100i64..100), 0..50)
    ) {
        let rt = Runtime::new();
        let col = rt.create_collection::<i64>();
        let sum = col.reduce(&rt, |elems| -> i64 { elems.iter().sum() });

        for &(is_insert, val) in &ops {
            if is_insert {
                col.insert(&rt, val);
            } else {
                col.delete(&rt, &val);
            }
        }

        let expected: i64 = oracle_elements(&ops).into_iter().sum();
        let actual = rt.get(sum);
        prop_assert_eq!(actual, expected);
    }
}
