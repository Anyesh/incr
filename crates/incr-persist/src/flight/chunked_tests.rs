use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use super::{Chunked, Control, Verdict};
use crate::{Error, Result, ValueScope};

thread_local! {
    static IN_IDLE: Cell<bool> = const { Cell::new(false) };
    static IDLES: Cell<usize> = const { Cell::new(0) };
}

fn marking_idle(f: &mut dyn FnMut()) {
    IDLES.with(|idles| idles.set(idles.get() + 1));
    IN_IDLE.with(|flag| flag.set(true));
    f();
    IN_IDLE.with(|flag| flag.set(false));
}

fn idle_scope() -> Option<ValueScope> {
    Some(ValueScope::new(Arc::new(())).idling(marking_idle))
}

/// `len` items counted as they are pulled, failing at `fail_at`.
fn source(
    len: u32,
    fail_at: Option<u32>,
    pulls: &Rc<Cell<u32>>,
) -> impl Iterator<Item = Result<u32>> {
    let pulls = Rc::clone(pulls);
    let mut next = 0;
    std::iter::from_fn(move || {
        if next == len {
            return None;
        }
        pulls.set(pulls.get() + 1);
        let item = next;
        next += 1;
        if Some(item) == fail_at {
            next = len;
            return Some(Err(Error::Corrupt("pull failed")));
        }
        Some(Ok(item))
    })
}

fn chunked(
    len: u32,
    fail_at: Option<u32>,
) -> (
    Chunked<u32, impl Iterator<Item = Result<u32>>>,
    Arc<Control>,
) {
    let control = Arc::new(Control::default());
    let source = source(len, fail_at, &Rc::new(Cell::new(0)));
    let chunked = Chunked::new(source, Arc::clone(&control), idle_scope());
    (chunked, control)
}

/// A backend that pulls `take` items, or until the end, passing on an
/// `Err` unless it swallows them, and keeps what it pulled.
fn backend(
    records: &mut dyn Iterator<Item = Result<u32>>,
    take: Option<usize>,
    swallow: bool,
    kept: &mut Vec<u32>,
) -> Result<usize> {
    while take.is_none_or(|take| kept.len() < take) {
        match records.next() {
            None => break,
            Some(Ok(item)) => kept.push(item),
            Some(Err(_)) if swallow => {}
            Some(Err(error)) => return Err(error),
        }
    }
    Ok(kept.len())
}

fn flight(len: u32, fail_at: Option<u32>, take: Option<usize>, swallow: bool) -> Verdict<usize> {
    let (mut records, _) = chunked(len, fail_at);
    let result = backend(&mut records, take, swallow, &mut Vec::new());
    records.finish(result).0
}

#[test]
fn a_backend_that_pulls_to_the_end_commits_every_item_in_order() {
    let (mut records, _) = chunked(1000, None);
    let mut kept = Vec::new();
    let result = backend(&mut records, None, false, &mut kept);
    assert_eq!(kept, (0..1000).collect::<Vec<_>>());
    assert!(matches!(records.finish(result).0, Verdict::Committed(1000)));
}

#[test]
fn a_backend_that_returns_after_one_record_is_truncated() {
    assert!(matches!(
        flight(1000, None, Some(1), false),
        Verdict::Truncated
    ));
}

#[test]
fn a_backend_that_returns_after_exactly_one_chunk_is_truncated() {
    assert!(matches!(
        flight(1000, None, Some(256), false),
        Verdict::Truncated
    ));
    assert!(matches!(
        flight(257, None, Some(256), false),
        Verdict::Truncated
    ));
}

#[test]
fn a_backend_that_takes_every_record_without_the_final_none_commits() {
    for len in [256, 512, 768] {
        assert!(
            matches!(
                flight(len, None, Some(len as usize), false),
                Verdict::Committed(n) if n == len as usize
            ),
            "{len} records"
        );
    }
}

#[test]
fn a_backend_that_swallows_a_failed_pull_and_returns_ok_is_truncated() {
    assert!(matches!(
        flight(1000, Some(300), None, true),
        Verdict::Truncated
    ));
    assert!(matches!(
        flight(1000, Some(0), None, true),
        Verdict::Truncated
    ));
}

#[test]
fn a_failed_pull_reaches_the_backend_after_the_items_before_it() {
    let (mut records, _) = chunked(1000, Some(300));
    let mut kept = Vec::new();
    let result = backend(&mut records, None, false, &mut kept);
    assert_eq!(kept, (0..300).collect::<Vec<_>>());
    assert!(matches!(result, Err(Error::Corrupt("pull failed"))));
    assert!(records.next().is_none(), "nothing follows a handed-out Err");
    assert!(matches!(
        records.finish(result).0,
        Verdict::Failed(Error::Corrupt("pull failed"))
    ));
}

#[test]
fn a_failed_pull_the_backend_never_reached_truncates() {
    let (mut records, _) = chunked(1000, Some(10));
    let result = backend(&mut records, Some(5), false, &mut Vec::new());
    assert!(matches!(records.finish(result).0, Verdict::Truncated));
}

#[test]
fn a_cancel_at_a_pull_reaches_the_backend_as_an_error() {
    let (mut records, control) = chunked(1000, None);
    let mut kept = Vec::new();
    backend(&mut records, Some(10), false, &mut kept).unwrap();
    control.cancel();
    let result = backend(&mut records, None, false, &mut kept);
    assert_eq!(kept.len(), 10, "no buffered item follows the cancel");
    assert!(matches!(result, Err(Error::Cancelled)));
    assert!(matches!(records.finish(result).0, Verdict::Cancelled));
}

#[test]
fn an_abandon_after_the_last_item_is_handed_out_in_place_of_the_end() {
    let (mut records, control) = chunked(100, None);
    let mut kept = Vec::new();
    backend(&mut records, Some(100), false, &mut kept).unwrap();
    control.abandon();
    assert!(matches!(records.next(), Some(Err(Error::Cancelled))));
    assert!(records.next().is_none());
}

#[test]
fn a_backend_that_swallows_a_cancel_and_returns_ok_is_truncated() {
    let (mut records, control) = chunked(1000, None);
    control.cancel();
    let result = backend(&mut records, None, true, &mut Vec::new());
    assert!(matches!(records.finish(result).0, Verdict::Truncated));
}

#[test]
fn a_failing_backend_fails_the_flight() {
    let (mut records, _) = chunked(1000, None);
    backend(&mut records, Some(3), false, &mut Vec::new()).unwrap();
    let result: Result<usize> = Err(Error::ReadOnly);
    assert!(matches!(
        records.finish(result).0,
        Verdict::Failed(Error::ReadOnly)
    ));
}

#[test]
fn a_backend_that_fails_gets_no_extra_pull() {
    let calls = Rc::new(Cell::new(0));
    let counted = Rc::clone(&calls);
    let mut inner = source(256, None, &Rc::new(Cell::new(0)));
    let counting = std::iter::from_fn(move || {
        counted.set(counted.get() + 1);
        inner.next()
    });
    let mut records = Chunked::new(counting, Arc::new(Control::default()), None);
    backend(&mut records, Some(256), false, &mut Vec::new()).unwrap();
    let before = calls.get();
    let result: Result<usize> = Err(Error::ReadOnly);
    assert!(matches!(
        records.finish(result).0,
        Verdict::Failed(Error::ReadOnly)
    ));
    assert_eq!(calls.get(), before, "the source was pulled after a failure");
}

#[test]
fn pulls_run_inside_idle_at_most_256_per_refill() {
    let pulls = Rc::new(Cell::new(0));
    let control = Arc::new(Control::default());
    let checked = source(1000, None, &pulls).inspect(|_| {
        assert!(
            IN_IDLE.with(Cell::get),
            "a pull ran outside the value scope's idle"
        )
    });
    let mut records = Chunked::new(checked, control, idle_scope());
    let idles = IDLES.with(Cell::get);
    assert!(matches!(records.next(), Some(Ok(0))));
    assert!(!IN_IDLE.with(Cell::get));
    assert_eq!(IDLES.with(Cell::get), idles + 1);
    assert!((1..=256).contains(&pulls.get()), "{} pulls", pulls.get());
    for _ in 1..pulls.get() {
        records.next();
    }
    assert_eq!(IDLES.with(Cell::get), idles + 1, "one idle per refill");
    let before = pulls.get();
    records.next();
    assert!(pulls.get() > before);
    assert_eq!(IDLES.with(Cell::get), idles + 2);
    for _ in before + 1..pulls.get() {
        records.next();
    }
    let idles = IDLES.with(Cell::get);
    assert!(matches!(records.finish(Ok(())).0, Verdict::Truncated));
    assert_eq!(IDLES.with(Cell::get), idles + 1, "finish pulls inside idle");
}

#[test]
fn a_refill_stops_pulling_after_2_ms() {
    let pulls = Rc::new(Cell::new(0));
    let slow = source(1000, None, &pulls).inspect(|_| std::thread::sleep(Duration::from_millis(1)));
    let mut records = Chunked::new(slow, Arc::new(Control::default()), None);
    assert!(matches!(records.next(), Some(Ok(0))));
    assert!(pulls.get() < 100, "{} pulls in one refill", pulls.get());
}

/// A run whose first refill panicked at its first pull, leaving the
/// buffer empty, and whose source then ends.
fn panicked_refill() -> Chunked<u32, impl Iterator<Item = Result<u32>>> {
    let mut pulled = false;
    let source = std::iter::from_fn(move || {
        if !pulled {
            pulled = true;
            panic!("value code panicked");
        }
        None::<Result<u32>>
    });
    let mut records = Chunked::new(source, Arc::new(Control::default()), None);
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| records.next()));
    assert!(unwound.is_err());
    assert!(records.next().is_none());
    records
}

#[test]
fn a_refill_that_panicked_is_never_committed() {
    assert!(matches!(
        panicked_refill().finish(Ok(())).0,
        Verdict::Truncated
    ));
    let result: Result<()> = Err(Error::ReadOnly);
    assert!(matches!(
        panicked_refill().finish(result).0,
        Verdict::Failed(Error::ReadOnly)
    ));
}
