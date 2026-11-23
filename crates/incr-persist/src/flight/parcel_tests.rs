use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use super::Parcel;
use crate::ValueScope;

thread_local! {
    static IN_HOLD: Cell<bool> = const { Cell::new(false) };
    static HOLDS: Cell<usize> = const { Cell::new(0) };
    static INSIDE: Cell<usize> = const { Cell::new(0) };
    static OUTSIDE: Cell<usize> = const { Cell::new(0) };
}

fn marking_hold(f: &mut dyn FnMut()) {
    HOLDS.with(|holds| holds.set(holds.get() + 1));
    IN_HOLD.with(|flag| flag.set(true));
    let _clear = Clear;
    f();
}

struct Clear;

impl Drop for Clear {
    fn drop(&mut self) {
        IN_HOLD.with(|flag| flag.set(false));
    }
}

/// A value that counts where it was dropped.
struct Tracked(u32);

impl Drop for Tracked {
    fn drop(&mut self) {
        let counter = if IN_HOLD.with(Cell::get) {
            &INSIDE
        } else {
            &OUTSIDE
        };
        counter.with(|n| n.set(n.get() + 1));
    }
}

fn counts() -> (usize, usize, usize) {
    (
        HOLDS.with(Cell::get),
        INSIDE.with(Cell::get),
        OUTSIDE.with(Cell::get),
    )
}

fn parcel(len: u32) -> Parcel<Tracked> {
    let scope = ValueScope::new(Arc::new(())).holding(marking_hold);
    Parcel::new((0..len).map(Tracked).collect(), Some(scope))
}

#[test]
fn a_dropped_parcel_drops_every_value_inside_hold_256_at_a_time() {
    let parcel = parcel(600);
    let before = counts();
    drop(parcel);
    let (holds, inside, outside) = counts();
    assert_eq!(holds - before.0, 3);
    assert_eq!(inside - before.1, 600);
    assert_eq!(outside, before.2);
}

#[test]
fn take_batch_runs_on_256_values_in_order_inside_one_hold() {
    let mut parcel = parcel(600);
    let before = counts();
    let first = parcel.take_batch(|value| {
        assert!(IN_HOLD.with(Cell::get));
        value.0
    });
    assert_eq!(first, (0..256).collect::<Vec<_>>());
    let (holds, inside, outside) = counts();
    assert_eq!((holds - before.0, inside - before.1), (1, 256));
    assert_eq!(outside, before.2);
    assert_eq!(parcel.take_batch(|value| value.0).len(), 256);
    assert_eq!(
        parcel.take_batch(|value| value.0),
        (512..600).collect::<Vec<_>>()
    );
    assert!(parcel.is_empty());
    let holds = HOLDS.with(Cell::get);
    assert!(parcel.take_batch(|value| value.0).is_empty());
    drop(parcel);
    assert_eq!(
        HOLDS.with(Cell::get),
        holds,
        "an empty parcel takes no hold"
    );
}

#[test]
fn values_drop_inside_hold_while_a_panic_unwinds() {
    let before = counts();
    let unwound = panic::catch_unwind(|| {
        let _parcel = parcel(300);
        panic!("value code panicked");
    });
    assert!(unwound.is_err());
    let (_, inside, outside) = counts();
    assert_eq!(inside - before.1, 300);
    assert_eq!(outside, before.2);
}

#[test]
fn a_panic_inside_take_batch_drops_the_rest_inside_hold() {
    let mut parcel = parcel(300);
    let before = counts();
    let unwound = panic::catch_unwind(AssertUnwindSafe(|| {
        parcel.take_batch(|value| {
            if value.0 == 10 {
                panic!("encode panicked");
            }
        })
    }));
    assert!(unwound.is_err());
    drop(parcel);
    let (_, inside, outside) = counts();
    assert_eq!(inside - before.1, 300);
    assert_eq!(outside, before.2);
}

#[test]
fn append_and_reorder_move_values_without_dropping_them() {
    let mut first = parcel(3);
    first.append(parcel(2));
    first.push(Tracked(9));
    let before = counts();
    first
        .as_mut_slice()
        .sort_by_key(|value| std::cmp::Reverse(value.0));
    assert_eq!(counts(), before);
    assert_eq!(first.len(), 6);
    assert_eq!(first.take_batch(|value| value.0), vec![9, 2, 1, 1, 0, 0]);
}
