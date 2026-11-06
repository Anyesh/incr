use std::cell::Cell;

use super::tests::user_code_panics::failure;
use super::*;
use crate::cells::{Local, Shared};
use crate::value_code::CONTRACT_MESSAGE;

fn even(x: &i64) -> bool {
    x % 2 == 0
}

fn counts_the_bootstrap_and_every_delta<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let c = rt.create_collection::<i64>();
    for x in [1, 2, 3, 4] {
        c.insert(&rt, x);
    }
    let parts = c.count_partition(&rt, even);
    assert_eq!(rt.get(parts), (2, 2));

    c.insert(&rt, 6);
    c.insert(&rt, 7);
    c.insert(&rt, 7);
    assert_eq!(rt.get(parts), (3, 4));

    assert!(c.delete(&rt, &2));
    assert!(c.delete(&rt, &7));
    assert!(!c.delete(&rt, &100));
    assert_eq!(rt.get(parts), (2, 3));

    c.replace(&rt, Some(&1), 8);
    assert_eq!(rt.get(parts), (3, 2));
}

#[test]
fn counts_the_bootstrap_and_every_delta_local() {
    counts_the_bootstrap_and_every_delta::<Local>();
}

#[test]
fn counts_the_bootstrap_and_every_delta_shared() {
    counts_the_bootstrap_and_every_delta::<Shared>();
}

fn sums_to_count_over_a_derived_collection<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let c = rt.create_collection::<i64>();
    c.extend(&rt, 0..500);
    let kept = c.filter(&rt, |x| x % 3 != 0);
    let parts = kept.count_partition(&rt, even);
    let count = kept.count(&rt);
    let check = |rt: &Runtime<C>| {
        let values = kept.values(rt);
        let evens = values.iter().filter(|x| even(x)).count() as u64;
        let (matching, other) = rt.get(parts);
        assert_eq!((matching, other), (evens, values.len() as u64 - evens));
        assert_eq!(matching + other, rt.get(count));
    };
    check(&rt);
    c.extend(&rt, 500..900);
    for x in (0..900).step_by(7) {
        c.delete(&rt, &x);
    }
    check(&rt);
}

#[test]
fn sums_to_count_over_a_derived_collection_local() {
    sums_to_count_over_a_derived_collection::<Local>();
}

#[test]
fn sums_to_count_over_a_derived_collection_shared() {
    sums_to_count_over_a_derived_collection::<Shared>();
}

#[test]
fn a_predicate_panic_counts_nothing_twice() {
    thread_local! {
        static PANIC_ON: Cell<i64> = const { Cell::new(-1) };
    }
    let rt: Runtime<Local> = Runtime::new();
    let c = rt.create_collection::<i64>();
    let parts = c.count_partition(&rt, |x: &i64| {
        assert_ne!(PANIC_ON.with(Cell::get), *x, "chosen predicate call");
        even(x)
    });
    assert_eq!(rt.get(parts), (0, 0));

    c.extend(&rt, [1, 2, 3, 4]);
    PANIC_ON.with(|p| p.set(3));
    let message = failure(|| {
        rt.get(parts);
    });
    assert!(message.is_some_and(|m| m.contains("chosen predicate call")));
    PANIC_ON.with(|p| p.set(-1));

    c.insert(&rt, 5);
    assert_eq!(rt.get(parts), (2, 3));
}

#[test]
fn the_predicate_runs_as_value_code() {
    let rt: Runtime<Local> = Runtime::new();
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let message = failure(|| {
        let _ = c.count_partition(&rt, |_| {
            value_code::check();
            true
        });
    });
    assert!(message.is_some_and(|m| m.contains(CONTRACT_MESSAGE)));
}
