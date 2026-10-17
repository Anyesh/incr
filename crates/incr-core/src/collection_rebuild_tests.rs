//! Bug (k): a user panic part-way through an operator's batch must not
//! leave a prefix of the batch applied for the retry to apply again. After
//! recovery every operator, and a consumer downstream of it, must read the
//! same as a fresh operator built over the same input.

use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::{Arc, Weak};
use std::time::Duration;

use super::tests::user_code_panics::{arm, disarm, failure, populated, sorted, tick, Bomb, Call};
use super::*;
use crate::cells::{Local, Shared};

const CALLS: [Call; 5] = [Call::Hash, Call::Eq, Call::Clone, Call::Drop, Call::Closure];

/// No batch below makes this many calls of one kind, so reaching it means
/// the armed call kept firing.
const MAX_NTH: usize = 2000;

struct Sources<C: Cells> {
    left: IncrCollection<Bomb, C>,
    right: IncrCollection<Bomb, C>,
}

/// An operator under test: the node whose compute runs its batch, the node
/// of an intermediate operator it reads (forced again by the park test
/// after feeding the sources), and a reader of its output and of a
/// downstream consumer built alongside it.
struct Probe<C: Cells> {
    node: Incr<u64>,
    upstream: Option<Incr<u64>>,
    read: Reader<C>,
}

type Build<C> = fn(&Runtime<C>, &Sources<C>) -> Probe<C>;
type Reader<C> = Box<dyn Fn(&Runtime<C>) -> String>;

fn filter_op<C: Cells>(rt: &Runtime<C>, s: &Sources<C>) -> Probe<C> {
    let out = s.left.filter(rt, |b| {
        tick(Call::Closure);
        b.0 % 3 != 0
    });
    let down = out.filter(rt, |_| true);
    Probe {
        node: out.version_node,
        upstream: None,
        read: Box::new(move |rt| {
            format!(
                "{:?} / {:?}",
                sorted(out.values(rt)),
                sorted(down.values(rt))
            )
        }),
    }
}

fn map_op<C: Cells>(rt: &Runtime<C>, s: &Sources<C>) -> Probe<C> {
    let out = s.left.map(rt, |b| {
        tick(Call::Closure);
        Bomb(b.0 / 2)
    });
    let down = out.filter(rt, |_| true);
    Probe {
        node: out.version_node,
        upstream: None,
        read: Box::new(move |rt| {
            format!(
                "{:?} / {:?}",
                sorted(out.values(rt)),
                sorted(down.values(rt))
            )
        }),
    }
}

fn aggregate_op<C: Cells>(rt: &Runtime<C>, s: &Sources<C>) -> Probe<C> {
    let sum = s.left.aggregate(
        rt,
        0_u64,
        |b| {
            tick(Call::Closure);
            b.0 as u64
        },
        |x, y| {
            tick(Call::Closure);
            x + y
        },
    );
    Probe {
        node: sum,
        upstream: None,
        read: Box::new(move |rt| rt.get(sum).to_string()),
    }
}

fn operators<C: Cells>() -> [(&'static str, Build<C>); 3] {
    [
        ("filter", filter_op::<C>),
        ("map", map_op::<C>),
        ("aggregate", aggregate_op::<C>),
    ]
}

fn sources<C: Cells>(rt: &Runtime<C>) -> Sources<C> {
    Sources {
        left: populated(rt, &[1, 2, 3, 4, 5, 6]),
        right: populated(rt, &[3, 4, 9]),
    }
}

/// The batch every run applies before forcing the operator: inserts,
/// deletes and a duplicate on both sides.
fn feed_batch<C: Cells>(rt: &Runtime<C>, s: &Sources<C>) {
    s.left.insert(rt, Bomb(7));
    s.left.insert(rt, Bomb(4));
    s.left.delete(rt, &Bomb(2));
    s.left.insert(rt, Bomb(8));
    s.left.delete(rt, &Bomb(6));
    s.right.insert(rt, Bomb(5));
    s.right.delete(rt, &Bomb(9));
    s.right.insert(rt, Bomb(3));
}

/// A later change, which also moves an operator left Failed by the panic
/// to recompute.
fn feed_later<C: Cells>(rt: &Runtime<C>, s: &Sources<C>) {
    s.left.insert(rt, Bomb(10));
    s.right.insert(rt, Bomb(11));
}

/// Forces the operator with the `nth` call of `call` panicking, then
/// checks that after recovery it reads the same as a fresh operator.
/// Returns whether the forced run panicked, or the mismatch.
fn run<C: Cells>(build: Build<C>, call: Call, nth: usize) -> Result<bool, String> {
    let rt: Runtime<C> = Runtime::new();
    let s = sources(&rt);
    let probe = build(&rt, &s);
    let _ = (probe.read)(&rt);
    feed_batch(&rt, &s);

    arm(call, nth);
    let panicked = failure(|| {
        rt.get(probe.node);
    })
    .is_some();
    disarm();

    feed_later(&rt, &s);
    let mut got = String::new();
    if let Some(e) = failure(|| got = (probe.read)(&rt)) {
        return Err(format!("reading after recovery panicked: {e}"));
    }
    let fresh = (build(&rt, &s).read)(&rt);
    if got == fresh {
        Ok(panicked)
    } else {
        Err(format!("recovered {got}, fresh {fresh}"))
    }
}

fn operators_recover_from_user_panics<C: Cells>() {
    let mut failures = Vec::new();
    for (name, build) in operators::<C>() {
        for call in CALLS {
            for nth in 1.. {
                if nth == MAX_NTH {
                    failures.push(format!("{name}, {call:?}: never ran out of calls"));
                    break;
                }
                match run(build, call, nth) {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(e) => failures.push(format!("{name}, {call:?} call {nth}: {e}")),
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} runs disagreed with a fresh operator:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_operators_recover_from_user_panics() {
    operators_recover_from_user_panics::<Local>();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_operators_recover_from_user_panics() {
    operators_recover_from_user_panics::<Shared>();
}

const PARK_TIMEOUT: Duration = Duration::from_secs(10);

enum Event {
    Parked,
    Done,
}

/// Parks every rebuild of one operator just after it read an upstream
/// snapshot, and feeds the sources while it is parked. Returns whether the
/// forced run panicked and whether a rebuild parked, or the mismatch.
fn run_parked(build: Build<Shared>, call: Call, nth: usize) -> Result<(bool, bool), String> {
    let rt: Runtime<Shared> = Runtime::new();
    let (event_tx, event_rx) = mpsc::channel::<Event>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let armed = Arc::new(AtomicBool::new(false));
    {
        let armed = Arc::clone(&armed);
        let event_tx = std::sync::Mutex::new(event_tx.clone());
        let release_rx = std::sync::Mutex::new(release_rx);
        let installed = rt.park_hook.set(Box::new(move |point| {
            if point == ParkPoint::RebuildAfterSnapshot && armed.load(MemOrdering::SeqCst) {
                event_tx.lock().unwrap().send(Event::Parked).unwrap();
                release_rx
                    .lock()
                    .unwrap()
                    .recv_timeout(PARK_TIMEOUT)
                    .expect("never released");
            }
        }));
        assert!(installed.is_ok());
    }

    let s = sources(&rt);
    let probe = build(&rt, &s);
    let _ = (probe.read)(&rt);
    feed_batch(&rt, &s);

    arm(call, nth);
    let panicked = failure(|| {
        rt.get(probe.node);
    })
    .is_some();
    disarm();
    feed_later(&rt, &s);

    armed.store(true, MemOrdering::SeqCst);
    let mut parked = false;
    std::thread::scope(|scope| {
        let node = probe.node;
        let rt = &rt;
        scope.spawn(move || {
            let _ = failure(|| {
                rt.get(node);
            });
            event_tx.send(Event::Done).unwrap();
        });
        let mut next = 20;
        loop {
            match event_rx.recv_timeout(PARK_TIMEOUT).expect("no event") {
                Event::Done => break,
                Event::Parked => {
                    parked = true;
                    s.left.insert(rt, Bomb(next));
                    s.right.insert(rt, Bomb(next + 1));
                    next += 2;
                    if let Some(up) = probe.upstream {
                        rt.get(up);
                    }
                    release_tx.send(()).unwrap();
                }
            }
        }
    });
    armed.store(false, MemOrdering::SeqCst);

    let got = (probe.read)(&rt);
    let fresh = (build(&rt, &s).read)(&rt);
    if got == fresh {
        Ok((panicked, parked))
    } else {
        Err(format!("recovered {got}, fresh {fresh}"))
    }
}

/// An insert landing while a rebuild is parked between its snapshot and
/// the rest of its run is either in the snapshot or pending after it.
#[test]
#[cfg_attr(miri, ignore)]
fn rebuild_misses_no_insert_landing_after_its_snapshot() {
    let mut failures = Vec::new();
    for (name, build) in operators::<Shared>() {
        let mut parked_once = false;
        'calls: for call in CALLS {
            for nth in 1..MAX_NTH {
                match run_parked(build, call, nth) {
                    Ok((_, true)) => {
                        parked_once = true;
                        break 'calls;
                    }
                    Ok((true, false)) => {}
                    Ok((false, false)) => break,
                    Err(e) => {
                        failures.push(format!("{name}, {call:?} call {nth}: {e}"));
                        break 'calls;
                    }
                }
            }
        }
        if !parked_once && failures.iter().all(|f| !f.starts_with(name)) {
            failures.push(format!("{name}: no panic led to a rebuild"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} operators missed an insert:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// `combine` runs under the aggregate's state lock, which only that
/// aggregate's compute takes, so it may read other queries.
fn sum_whose_combine_reads(
    rt: &Arc<Runtime<Shared>>,
    c: &IncrCollection<i64, Shared>,
    query: Incr<u64>,
) -> Incr<i64> {
    let weak: Weak<Runtime<Shared>> = Arc::downgrade(rt);
    c.aggregate(
        rt,
        0,
        |x| *x,
        move |a, b| {
            let rt = weak.upgrade().expect("runtime alive");
            let _ = rt.get(query);
            a + b
        },
    )
}

fn assert_combine_can_read(
    make_query: fn(&Runtime<Shared>, &IncrCollection<i64, Shared>) -> Incr<u64>,
) {
    let rt = Arc::new(Runtime::<Shared>::new());
    let c = rt.create_collection::<i64>();
    c.insert(&rt, 1);
    let query = make_query(&rt, &c);
    let sum = sum_whose_combine_reads(&rt, &c, query);
    assert_eq!(rt.get(sum), 1);
    for x in 2..=20 {
        c.insert(&rt, x);
    }
    assert_eq!(rt.get(sum), 210);
    c.delete(&rt, &7);
    assert_eq!(rt.get(sum), 203);
}

#[test]
fn aggregate_combine_reads_a_query_over_a_filter() {
    assert_combine_can_read(|rt, c| c.filter(rt, |x| x % 2 == 0).count(rt));
}

#[test]
fn aggregate_combine_reads_a_query_over_another_aggregate() {
    assert_combine_can_read(|rt, c| c.aggregate(rt, 0_u64, |x| *x as u64, |a, b| *a.max(b)));
}
