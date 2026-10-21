//! A type's cycle fallback ([`Runtime::set_cycle_fallback`]) used by
//! queries created with [`CycleStrategy::TypeFallback`]: each case runs a
//! graph twice, once with a per-node fallback on every such member and
//! once with the type's, and the two runs land the same values, states
//! and cycles. Each case runs under `Local` and `Shared`, unbounded and at
//! frame budget 1.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{Cycle, CycleStrategy, Fixpoint, QueryOpts, Replay};
use crate::handle::Incr;
use crate::state::NodeState;
use crate::table::NodeTable;

macro_rules! local_and_shared {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Kind {
    /// A Fallback: its own in the per-node run, the type's in the other.
    Fallback,
    Fixpoint,
    NoStrategy,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mode {
    PerNode,
    PerType,
}

/// What member `i` of a cycle lands as a fallback.
fn fallback_value(i: usize, cycle: &Cycle) -> u64 {
    1000 * (i as u64 + 1) + cycle.len() as u64
}

/// Member slots to ring indices, filled as members are created, so the
/// type's fallback can tell which member it was handed.
type Members = Arc<Mutex<HashMap<u32, usize>>>;

type Table<C> = NodeTable<usize, u64, C>;

/// `(member, handle, cycle)` for every call of the type's fallback.
type Calls = Arc<Mutex<Vec<(usize, Incr<u64>, Cycle)>>>;

type Compute<C> = Box<dyn Fn(&Runtime<C>) -> u64 + Send + Sync>;

struct Ring<C: Cells> {
    rt: Runtime<C>,
    table: Arc<Table<C>>,
    members: Members,
    calls: Calls,
}

fn opts(kinds: &[Kind], i: usize, mode: Mode) -> QueryOpts<u64> {
    let cycle = match (kinds[i], mode) {
        (Kind::Fallback, Mode::PerNode) => {
            CycleStrategy::fallback(move |cycle: &Cycle| fallback_value(i, cycle))
        }
        (Kind::Fallback, Mode::PerType) => CycleStrategy::TypeFallback,
        (Kind::Fixpoint, _) => CycleStrategy::Fixpoint(Fixpoint::new(|| 0_u64)),
        (Kind::NoStrategy, _) => CycleStrategy::None,
    };
    QueryOpts {
        cycle,
        replay: Replay::Allowed,
    }
}

/// Member `i` of a ring: it reads member `i + 1` through the table, which
/// creates it on a miss, and lands one more, capped so a Fixpoint ring
/// converges.
fn member<C: Cells>(
    table: Arc<Table<C>>,
    members: Members,
    kinds: Arc<[Kind]>,
    i: usize,
    mode: Mode,
) -> Compute<C>
where
    Table<C>: Send + Sync,
{
    Box::new(move |rt| {
        let next = (i + 1) % kinds.len();
        let h = node(rt, &table, &members, &kinds, next, mode);
        (rt.get(h) + 1).min(50)
    })
}

fn node<C: Cells>(
    rt: &Runtime<C>,
    table: &Arc<Table<C>>,
    members: &Members,
    kinds: &Arc<[Kind]>,
    i: usize,
    mode: Mode,
) -> Incr<u64>
where
    Table<C>: Send + Sync,
{
    let h = table.get_or_insert_query_with(
        rt,
        i,
        member(table.clone(), members.clone(), kinds.clone(), i, mode),
        opts(kinds, i, mode),
    );
    if rt.is_alive(h) {
        members.lock().unwrap().insert(h.slot(), i);
    }
    h
}

fn ring<C: Cells>(mode: Mode, frames: Option<u32>) -> Ring<C> {
    let rt = Runtime::<C>::new();
    rt.set_max_frames(frames.unwrap_or(u32::MAX));
    rt.set_placeholder(0_u64);
    let members: Members = Arc::default();
    let calls: Calls = Arc::default();
    if mode == Mode::PerType {
        let members = members.clone();
        let calls = calls.clone();
        rt.set_cycle_fallback(move |cycle: &Cycle, h: Incr<u64>| {
            let i = members.lock().unwrap()[&h.slot()];
            calls.lock().unwrap().push((i, h, cycle.clone()));
            fallback_value(i, cycle)
        });
    }
    Ring {
        table: Arc::new(rt.create_table()),
        rt,
        members,
        calls,
    }
}

type Landed = Vec<(Result<u64, Vec<u32>>, NodeState, Option<Vec<u32>>)>;

/// Enters the ring at `entry` with `try_get`, then reads every member.
fn observe<C: Cells>(r: &Ring<C>, kinds: &Arc<[Kind]>, entry: usize, mode: Mode) -> Landed
where
    Table<C>: Send + Sync,
{
    let slots = |c: &Cycle| {
        c.participants()
            .iter()
            .map(|p| p.slot())
            .collect::<Vec<_>>()
    };
    let handles: Vec<Incr<u64>> = (0..kinds.len())
        .map(|i| {
            node(
                &r.rt,
                &r.table,
                &r.members,
                kinds,
                (entry + i) % kinds.len(),
                mode,
            )
        })
        .collect();
    handles
        .iter()
        .map(|&h| {
            let v = r.rt.try_get(h).map_err(|c| slots(&c));
            (v, r.rt.state_of(h), r.rt.cycle_of(h).as_ref().map(slots))
        })
        .collect()
}

const RINGS: &[&[Kind]] = &[
    &[Kind::Fallback],
    &[Kind::Fallback, Kind::Fallback, Kind::Fallback],
    &[Kind::Fallback, Kind::NoStrategy, Kind::Fallback],
    &[Kind::Fallback, Kind::Fixpoint, Kind::NoStrategy],
    &[Kind::NoStrategy, Kind::Fallback],
];

/// Every ring, entered at every member, lands the same under the type's
/// fallback as under per-node fallbacks, and the type's fallback is
/// handed each member's own handle and a cycle that contains it.
fn type_fallback_lands_as_per_node_fallbacks<C: Cells>()
where
    Table<C>: Send + Sync,
{
    for kinds in RINGS {
        let kinds: Arc<[Kind]> = Arc::from(*kinds);
        for entry in 0..kinds.len() {
            for frames in [None, Some(1)] {
                let per_node = ring::<C>(Mode::PerNode, frames);
                let expected = observe(&per_node, &kinds, entry, Mode::PerNode);
                let per_type = ring::<C>(Mode::PerType, frames);
                let got = observe(&per_type, &kinds, entry, Mode::PerType);
                assert_eq!(
                    got, expected,
                    "{kinds:?} entered at {entry}, frames {frames:?}"
                );
                let calls = per_type.calls.lock().unwrap();
                assert!(!calls.is_empty(), "{kinds:?}: the type's fallback ran");
                for (i, h, cycle) in calls.iter() {
                    assert_eq!(kinds[*i], Kind::Fallback);
                    assert_eq!(per_type.table.get(&per_type.rt, i), Some(*h));
                    assert!(cycle.contains(*h));
                }
            }
        }
    }
}

/// Two threads entering a ring at different members land what one
/// thread does.
#[test]
fn shared_type_fallback_under_two_entries() {
    let kinds: Arc<[Kind]> = Arc::from(&[Kind::Fallback, Kind::Fallback, Kind::Fallback][..]);
    let per_node = ring::<Shared>(Mode::PerNode, None);
    let expected = observe(&per_node, &kinds, 0, Mode::PerNode);
    let r = ring::<Shared>(Mode::PerType, None);
    let handles: Vec<Incr<u64>> = (0..kinds.len())
        .map(|i| node(&r.rt, &r.table, &r.members, &kinds, i, Mode::PerType))
        .collect();
    std::thread::scope(|s| {
        for &h in &handles[..2] {
            let rt = &r.rt;
            s.spawn(move || rt.try_get(h).unwrap());
        }
    });
    assert_eq!(observe(&r, &kinds, 0, Mode::PerType), expected);
}

/// A query with no strategy has none even when its type has a fallback.
fn none_ignores_the_type_fallback<C: Cells>() {
    let rt = Runtime::<C>::new();
    rt.set_cycle_fallback(|_: &Cycle, _: Incr<u64>| 7_u64);
    let me = Arc::new(std::sync::OnceLock::<Incr<u64>>::new());
    let q = {
        let me = me.clone();
        rt.create_query(move |rt| rt.get(*me.get().unwrap()) + 1)
    };
    me.set(q).unwrap();
    assert_eq!(rt.try_get(q).map_err(|c| c.len()), Err(1));
}

/// A deleted type-fallback query's recycled slot, now a query of another
/// type with no strategy, takes no fallback.
fn a_recycled_slot_has_no_type_fallback<C: Cells>() {
    let rt = Runtime::<C>::new();
    rt.set_cycle_fallback(|_: &Cycle, _: Incr<u64>| 7_u64);
    let old = rt.create_query_with(
        |_| 1_u64,
        QueryOpts {
            cycle: CycleStrategy::TypeFallback,
            ..Default::default()
        },
    );
    rt.delete_node(old);
    let me = Arc::new(std::sync::OnceLock::<Incr<i64>>::new());
    let q = {
        let me = me.clone();
        rt.create_query(move |rt| rt.get(*me.get().unwrap()) + 1)
    };
    assert_eq!(q.slot(), old.slot());
    me.set(q).unwrap();
    assert!(rt.try_get(q).is_err());
}

fn type_fallback_without_one_registered_panics<C: Cells>() {
    let rt = Runtime::<C>::new();
    let err = catch_unwind(AssertUnwindSafe(|| {
        rt.create_query_with(
            |_| 1_u64,
            QueryOpts {
                cycle: CycleStrategy::TypeFallback,
                ..Default::default()
            },
        )
    }))
    .unwrap_err();
    let message = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(message.contains("set_cycle_fallback"), "{message}");
}

fn a_second_type_fallback_panics<C: Cells>() {
    let rt = Runtime::<C>::new();
    rt.set_cycle_fallback(|_: &Cycle, _: Incr<u64>| 1_u64);
    let err = catch_unwind(AssertUnwindSafe(|| {
        rt.set_cycle_fallback(|_: &Cycle, _: Incr<u64>| 2_u64)
    }));
    assert!(err.is_err());
}

local_and_shared!(
    type_fallback_lands_as_per_node_fallbacks,
    none_ignores_the_type_fallback,
    a_recycled_slot_has_no_type_fallback,
    type_fallback_without_one_registered_panics,
    a_second_type_fallback_panics,
);
