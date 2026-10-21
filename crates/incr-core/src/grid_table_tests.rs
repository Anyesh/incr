use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use super::{GridKey, GridTable, PAGE_ROWS};
use crate::cells::{Cells, Shared};
use crate::cycle::{Cycle, CycleStrategy, QueryOpts, Replay};
use crate::handle::Incr;
use crate::runtime::Runtime;

fn key(sheet: u32, row: u32, column: u32) -> GridKey {
    GridKey::new(sheet, row, column)
}

type Built<C> = (fn(&Runtime<C>) -> i64, QueryOpts<i64>);

fn replayable<T>() -> QueryOpts<T> {
    QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    }
}

fn panics_with(message: &str, f: impl FnOnce()) {
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    let got = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(got.contains(message), "expected {message:?}, got {got:?}");
}

macro_rules! both {
    ($($name:ident),* $(,)?) => {$(
        mod $name {
            #[test]
            fn local() {
                super::$name::<crate::cells::Local>(&crate::runtime::Runtime::new());
            }

            #[test]
            fn shared() {
                super::$name::<crate::cells::Shared>(&crate::runtime::Runtime::new());
            }
        }
    )*};
}

both!(
    grid_a_hit_returns_the_first_node,
    grid_keys_on_other_pages_columns_and_sheets_are_distinct,
    grid_a_deleted_node_is_a_miss_and_is_replaced,
    grid_computes_create_their_nodes_through_the_table,
    grid_lookups_and_inserts_refuse_a_compute,
    grid_another_runtime_is_refused,
    grid_a_dead_or_recycled_entry_is_a_miss,
    grid_a_key_out_of_range_is_refused,
    grid_a_doomed_frame_refuses_a_key_out_of_range,
    grid_a_doomed_miss_returns_a_pending_handle_and_creates_nothing,
    grid_insert_inputs_enters_every_node,
    grid_insert_queries_watched_watches_every_node,
    grid_an_insert_with_a_live_key_creates_nothing,
    grid_an_insert_with_a_repeated_key_creates_nothing,
    grid_an_insert_replaces_a_deleted_node,
    grid_type_fallback_queries_need_and_use_their_types_fallback,
    grid_a_lazy_query_is_built_only_on_a_miss,
);

fn grid_a_hit_returns_the_first_node<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_grid_table::<i64>();
    let k = key(0, 5, 2);
    assert_eq!(t.get(rt, k), None);
    let h = t.get_or_insert_input(rt, k, || 10);
    let ran = Cell::new(false);
    let again = t.get_or_insert_input(rt, k, || {
        ran.set(true);
        20
    });
    assert_eq!(h, again);
    assert!(!ran.get(), "a hit does not run init");
    assert_eq!(rt.get(h), 10);
    let q = t.get_or_insert_query(rt, k, |_| 30);
    assert_eq!(q, h, "a hit returns the entry whichever call created it");
    assert_eq!(t.get(rt, k), Some(h));
}

fn grid_keys_on_other_pages_columns_and_sheets_are_distinct<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_grid_table::<i64>();
    let mut keys = vec![
        key(0, 0, 0),
        key(0, PAGE_ROWS - 1, 0),
        key(0, PAGE_ROWS, 0),
        key(0, 100_000, 0),
        key(0, 0, 1),
        key(0, 0, 300),
        key(3, 0, 0),
        key(1, 7, 9),
    ];
    // Its directories hold about half a million entries, too slow to
    // initialize under Miri.
    if !cfg!(miri) {
        keys.push(key(
            GridKey::SHEETS - 1,
            GridKey::ROWS - 1,
            GridKey::COLUMNS - 1,
        ));
    }
    let handles: Vec<Incr<i64>> = keys
        .iter()
        .enumerate()
        .map(|(i, &k)| t.get_or_insert_input(rt, k, || i as i64))
        .collect();
    for (i, (&k, &h)) in keys.iter().zip(&handles).enumerate() {
        assert_eq!(t.get(rt, k), Some(h));
        assert_eq!(rt.get(h), i as i64);
    }
    assert_eq!(t.get(rt, key(0, 1, 0)), None);
    assert_eq!(t.get(rt, key(2, 0, 0)), None);
    assert_eq!(t.get(rt, key(0, 0, 2)), None);
    assert_eq!(rt.node_count(), keys.len());
}

fn grid_a_deleted_node_is_a_miss_and_is_replaced<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_grid_table::<i64>();
    let h = t.get_or_insert_input(rt, key(0, 1, 1), || 10);
    rt.delete_node(h);
    assert_eq!(t.get(rt, key(0, 1, 1)), None);
    let h2 = t.get_or_insert_input(rt, key(0, 1, 1), || 20);
    assert_ne!(h, h2);
    assert_eq!(rt.get(h2), 20);
    assert_eq!(t.get(rt, key(0, 1, 1)), Some(h2));
}

fn grid_computes_create_their_nodes_through_the_table<C: Cells>(rt: &Runtime<C>)
where
    GridTable<i64, C>: Send + Sync,
{
    let inputs = Arc::new(rt.create_grid_table::<i64>());
    let queries = Arc::new(rt.create_grid_table::<i64>());
    let n = rt.create_input(3_i64);
    let (i2, q2) = (inputs.clone(), queries.clone());
    let sum = rt.create_query(move |rt| {
        (0..rt.get(n) as u32)
            .map(|row| {
                let x = i2.get_or_insert_input(rt, key(0, row, 0), || 10 * i64::from(row));
                rt.get(q2.get_or_insert_query(rt, key(0, row, 1), move |rt| rt.get(x) + 1))
            })
            .sum::<i64>()
    });
    assert_eq!(rt.get(sum), 1 + 11 + 21);
    rt.set(n, 4);
    assert_eq!(rt.get(sum), 1 + 11 + 21 + 31);
    let x0 = inputs
        .get(rt, key(0, 0, 0))
        .expect("created by the compute");
    rt.set(x0, 100);
    assert_eq!(rt.get(sum), 101 + 11 + 21 + 31);
}

fn grid_lookups_and_inserts_refuse_a_compute<C: Cells>(rt: &Runtime<C>)
where
    GridTable<i64, C>: Send + Sync,
{
    let t = Arc::new(rt.create_grid_table::<i64>());
    t.get_or_insert_input(rt, key(0, 1, 1), || 1);
    type Call<C> = fn(&GridTable<i64, C>, &Runtime<C>);
    let calls: [(&str, Call<C>); 3] = [
        ("get", |t, rt| {
            t.get(rt, key(0, 1, 1));
        }),
        ("insert_inputs", |t, rt| {
            t.insert_inputs(rt, vec![(key(0, 2, 1), 2)]);
        }),
        ("insert_queries_watched", |t, rt| {
            t.insert_queries_watched(
                rt,
                vec![(key(0, 3, 1), |_: &Runtime<C>| 3, QueryOpts::default())],
            );
        }),
    ];
    for (name, call) in calls {
        let t2 = t.clone();
        let q = rt.create_query(move |rt| {
            call(&t2, rt);
            0_i64
        });
        panics_with(&format!("GridTable::{name} called during compute"), || {
            rt.get(q);
        });
    }
    let nodes = rt.node_count();
    assert_eq!(t.get(rt, key(0, 2, 1)), None);
    assert_eq!(rt.node_count(), nodes);
}

fn grid_another_runtime_is_refused<C: Cells>(rt: &Runtime<C>) {
    let other = Runtime::<C>::new();
    let t = other.create_grid_table::<i64>();
    panics_with("GridTable of runtime", || {
        t.get_or_insert_input(rt, key(0, 0, 0), || 1);
    });
    panics_with("GridTable of runtime", || {
        t.insert_inputs(rt, vec![(key(0, 0, 0), 1)]);
    });
}

/// A hit takes the entry's node only while that node is alive under the
/// entry's generation: a deleted node is a miss, and so is a slot another
/// node recycled, which is alive under a newer generation.
fn grid_a_dead_or_recycled_entry_is_a_miss<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_grid_table::<i64>();
    let (dead, recycled) = (key(0, 1, 1), key(0, 2, 1));
    let d = t.get_or_insert_input(rt, dead, || 10);
    let r = t.get_or_insert_query(rt, recycled, |_| 11);
    assert_eq!(rt.get(r), 11);
    rt.delete_node(r);
    let occupant = rt.create_input(99_i64);
    assert_eq!(occupant.slot(), r.slot(), "the occupant reuses the slot");
    rt.delete_node(d);
    assert_eq!(t.get(rt, dead), None);
    assert_eq!(t.get(rt, recycled), None);

    let ran = Cell::new(0);
    let d2 = t.get_or_insert_input(rt, dead, || {
        ran.set(ran.get() + 1);
        20
    });
    let r2 = t.get_or_insert_query(rt, recycled, |_| 21);
    assert_eq!(ran.get(), 1, "a dead entry is a miss");
    assert!(r2 != occupant && r2 != r, "a recycled entry is a miss");
    assert_eq!((rt.get(d2), rt.get(r2), rt.get(occupant)), (20, 21, 99));
    assert_eq!(t.get(rt, dead), Some(d2));
    assert_eq!(t.get(rt, recycled), Some(r2));
}

fn out_of_range_keys() -> [GridKey; 4] {
    [
        key(GridKey::SHEETS, 0, 0),
        key(0, GridKey::ROWS, 0),
        key(0, GridKey::ROWS + PAGE_ROWS - 1, 0),
        key(0, 0, GridKey::COLUMNS),
    ]
}

/// Whether `f` panicked with the out-of-range message.
fn refused_out_of_range<R>(f: impl FnOnce() -> R) -> bool {
    catch_unwind(AssertUnwindSafe(f)).is_err_and(|payload| {
        payload
            .downcast_ref::<String>()
            .is_some_and(|m| m.contains("out of range"))
    })
}

/// A lookup checks the key's range only on a miss, which an out-of-range
/// key always is, so it is refused whether or not the table has entries
/// at the edge of every bound.
fn grid_a_key_out_of_range_is_refused<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_grid_table::<i64>();
    let mut edges = vec![key(1, PAGE_ROWS, 1)];
    // Directories that large are too slow to initialize under Miri.
    if !cfg!(miri) {
        edges.extend([
            key(GridKey::SHEETS - 1, 0, 0),
            key(0, GridKey::ROWS - 1, 0),
            key(0, 0, GridKey::COLUMNS - 1),
        ]);
    }
    for populated in [false, true] {
        if populated {
            for &k in &edges {
                t.get_or_insert_input(rt, k, || 1);
            }
        }
        let nodes = rt.node_count();
        for k in out_of_range_keys() {
            let calls: [(&str, &dyn Fn()); 5] = [
                ("get_or_insert_input", &|| {
                    t.get_or_insert_input(rt, k, || 1);
                }),
                ("get_or_insert_query", &|| {
                    t.get_or_insert_query(rt, k, |_| 1);
                }),
                ("get_or_insert_query_lazy", &|| {
                    t.get_or_insert_query_lazy(rt, k, || -> Built<C> {
                        (|_| 1, QueryOpts::default())
                    });
                }),
                ("get", &|| {
                    t.get(rt, k);
                }),
                ("insert_inputs", &|| {
                    t.insert_inputs(rt, vec![(k, 1)]);
                }),
            ];
            for (call, f) in calls {
                assert!(
                    refused_out_of_range(f),
                    "{call} {k:?}, populated {populated}"
                );
            }
        }
        assert_eq!(rt.node_count(), nodes);
    }
}

/// The range check comes before the doomed frame's pending handle, so a
/// doomed compute's out-of-range key panics as it would undoomed.
fn grid_a_doomed_frame_refuses_a_key_out_of_range<C: Cells>(rt: &Runtime<C>)
where
    GridTable<i64, C>: Send + Sync,
{
    rt.set_placeholder::<i64>(-1);
    let gate = rt.doom_gate();
    let t = Arc::new(rt.create_grid_table::<i64>());
    t.get_or_insert_input(rt, key(0, 0, 0), || 1);
    let refused = Arc::new(AtomicU32::new(0));
    let doomed = Arc::new(AtomicU32::new(0));
    let (t2, count, pending) = (t.clone(), refused.clone(), doomed.clone());
    let q = rt.create_query_with(
        move |rt| {
            if rt.get(gate).0 {
                return 0;
            }
            let p = t2.get_or_insert_input(rt, key(0, 5, 0), || 0);
            pending.fetch_add(u32::from(!rt.is_alive(p)), Ordering::Relaxed);
            for k in out_of_range_keys() {
                let input = refused_out_of_range(|| t2.get_or_insert_input(rt, k, || 0));
                let query = refused_out_of_range(|| t2.get_or_insert_query(rt, k, |_| 0));
                count.fetch_add(u32::from(input) + u32::from(query), Ordering::Relaxed);
            }
            0_i64
        },
        replayable(),
    );
    let nodes = rt.node_count();
    assert_eq!(rt.get(q), 0);
    assert_eq!(doomed.load(Ordering::Relaxed), 1, "one doomed run");
    assert_eq!(refused.load(Ordering::Relaxed), 8);
    assert_eq!(rt.node_count(), nodes);
}

#[derive(Default)]
struct Doomed {
    hit: Option<Incr<i64>>,
    pending: Option<Incr<i64>>,
    query: Option<Incr<i64>>,
}

fn grid_a_doomed_miss_returns_a_pending_handle_and_creates_nothing<C: Cells>(rt: &Runtime<C>)
where
    GridTable<i64, C>: Send + Sync,
{
    rt.set_placeholder::<i64>(-1);
    let gate = rt.doom_gate();
    let t = Arc::new(rt.create_grid_table::<i64>());
    let present = t.get_or_insert_input(rt, key(0, 1, 0), || 10);
    let seen = Arc::new(Mutex::new(Doomed::default()));
    let (t2, out) = (t.clone(), seen.clone());
    let q = rt.create_query_with(
        move |rt| {
            if rt.get(gate).0 {
                return 0;
            }
            let mut out = out.lock().unwrap();
            out.hit = Some(t2.get_or_insert_input(rt, key(0, 1, 0), || 0));
            let p = t2.get_or_insert_input(rt, key(0, 2, 0), || 0);
            assert_eq!(rt.get(p), -1, "a pending handle reads the placeholder");
            assert!(!rt.is_alive(p));
            out.pending = Some(p);
            out.query = Some(t2.get_or_insert_query(rt, key(0, 3, 0), |_| 0));
            let lazy = t2.get_or_insert_query_lazy(rt, key(0, 4, 0), || -> Built<C> {
                unreachable!("a doomed miss builds nothing")
            });
            assert_eq!(lazy, p);
            0_i64
        },
        replayable(),
    );
    let nodes = rt.node_count();
    assert_eq!(rt.get(q), 0);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.hit, Some(present), "a hit returns the real handle");
    assert_eq!(
        seen.pending, seen.query,
        "pending handles of one runtime are equal"
    );
    assert_eq!(rt.node_count(), nodes, "a doomed miss creates nothing");
    assert_eq!(t.get(rt, key(0, 2, 0)), None, "and enters nothing");
    assert_eq!(t.get(rt, key(0, 3, 0)), None);
    assert_eq!(t.get(rt, key(0, 4, 0)), None);
}

fn grid_insert_inputs_enters_every_node<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_grid_table::<i64>();
    let entries: Vec<(GridKey, i64)> = (0..200)
        .map(|row| (key(0, row, 4), i64::from(row)))
        .collect();
    let handles = t.insert_inputs(rt, entries.clone());
    assert_eq!(handles.len(), entries.len());
    for ((k, v), h) in entries.iter().zip(&handles) {
        assert_eq!(t.get(rt, *k), Some(*h));
        assert_eq!(rt.get(*h), *v);
        assert_eq!(t.get_or_insert_input(rt, *k, || -1), *h);
    }
    assert_eq!(rt.node_count(), entries.len());
    rt.set(handles[7], 70);
    assert_eq!(rt.get(handles[7]), 70);
    assert_eq!(t.insert_inputs(rt, Vec::new()), Vec::new());
}

fn grid_insert_queries_watched_watches_every_node<C: Cells>(rt: &Runtime<C>) {
    let inputs = rt.create_grid_table::<i64>();
    let queries = rt.create_grid_table::<i64>();
    let x = inputs.insert_inputs(rt, (0..3).map(|row| (key(0, row, 0), 1)).collect());
    let entries = x
        .iter()
        .enumerate()
        .map(|(row, &x)| {
            (
                key(0, row as u32, 1),
                move |rt: &Runtime<C>| rt.get(x) * 10,
                QueryOpts::default(),
            )
        })
        .collect();
    let q = queries.insert_queries_watched(rt, entries);
    let mut drained: Vec<_> = rt.drain_dirtied_watched();
    drained.sort_by_key(|h| h.slot());
    let mut expected: Vec<_> = q.iter().map(|h| h.erase()).collect();
    expected.sort_by_key(|h| h.slot());
    assert_eq!(drained, expected, "a new query is reported at once");
    for (row, &h) in q.iter().enumerate() {
        assert_eq!(queries.get(rt, key(0, row as u32, 1)), Some(h));
        assert_eq!(rt.get(h), 10);
    }
    assert!(rt.drain_dirtied_watched().is_empty());
    rt.set(x[1], 2);
    assert_eq!(rt.drain_dirtied_watched(), vec![q[1].erase()]);
    assert_eq!(rt.get(q[1]), 20);
}

fn grid_an_insert_with_a_live_key_creates_nothing<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_grid_table::<i64>();
    let present = t.get_or_insert_input(rt, key(0, 5, 0), || 5);
    let nodes = rt.node_count();
    panics_with("has a live node", || {
        t.insert_inputs(rt, vec![(key(0, 4, 0), 4), (key(0, 5, 0), 50)]);
    });
    assert_eq!(rt.node_count(), nodes);
    assert_eq!(
        t.get(rt, key(0, 4, 0)),
        None,
        "the claim on the first key is undone"
    );
    assert_eq!(t.get(rt, key(0, 5, 0)), Some(present));
    let h = t.insert_inputs(rt, vec![(key(0, 4, 0), 4)]);
    assert_eq!(t.get(rt, key(0, 4, 0)), Some(h[0]));
}

fn grid_an_insert_with_a_repeated_key_creates_nothing<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_grid_table::<i64>();
    fn one<C: Cells>(_: &Runtime<C>) -> i64 {
        1
    }
    panics_with("appears twice", || {
        t.insert_queries_watched(
            rt,
            [key(0, 1, 0), key(0, 2, 0), key(0, 1, 0)]
                .into_iter()
                .map(|k| (k, one::<C>, QueryOpts::default()))
                .collect(),
        );
    });
    assert_eq!(rt.node_count(), 0);
    assert!(rt.drain_dirtied_watched().is_empty());
    assert_eq!(t.get(rt, key(0, 1, 0)), None);
    assert_eq!(t.get(rt, key(0, 2, 0)), None);
    let h = t.get_or_insert_input(rt, key(0, 1, 0), || 1);
    assert_eq!(t.get(rt, key(0, 1, 0)), Some(h));
}

fn grid_an_insert_replaces_a_deleted_node<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_grid_table::<i64>();
    let old = t.get_or_insert_input(rt, key(0, 0, 0), || 1);
    rt.delete_node(old);
    let new = t.insert_inputs(rt, vec![(key(0, 0, 0), 2)]);
    assert_eq!(t.get(rt, key(0, 0, 0)), Some(new[0]));
    assert_eq!(rt.get(new[0]), 2);
}

#[test]
fn grid_shared_racing_misses_agree_on_one_node() {
    let rt = Arc::new(Runtime::<Shared>::new());
    let t = Arc::new(rt.create_grid_table::<i64>());
    let rows = if cfg!(miri) { 8 } else { 2_000 };
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let (rt, t) = (rt.clone(), t.clone());
            std::thread::spawn(move || {
                (0..rows)
                    .map(|row| t.get_or_insert_input(&rt, key(0, row, row % 3), || i64::from(row)))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let seen: Vec<Vec<Incr<i64>>> = threads.into_iter().map(|h| h.join().unwrap()).collect();
    for other in &seen[1..] {
        assert_eq!(other, &seen[0]);
    }
    assert_eq!(rt.node_count(), rows as usize);
    for (row, &h) in seen[0].iter().enumerate() {
        assert_eq!(rt.get(h), row as i64);
    }
}

fn grid_type_fallback_queries_need_and_use_their_types_fallback<C: Cells>(rt: &Runtime<C>)
where
    GridTable<i64, C>: Send + Sync,
{
    fn type_fallback() -> QueryOpts<i64> {
        QueryOpts {
            cycle: CycleStrategy::TypeFallback,
            ..Default::default()
        }
    }
    let t = Arc::new(rt.create_grid_table::<i64>());
    panics_with("set_cycle_fallback", || {
        t.insert_queries_watched(
            rt,
            vec![(key(0, 9, 0), |_: &Runtime<C>| 1, type_fallback())],
        );
    });
    assert_eq!(rt.node_count(), 0);
    rt.set_cycle_fallback(|_: &Cycle, _: Incr<i64>| 7_i64);
    let reads = |row: u32| {
        let t = t.clone();
        move |rt: &Runtime<C>| {
            let next = t.get_or_insert_query_with(rt, key(0, row, 0), |_| 0, type_fallback());
            rt.get(next) + 1
        }
    };
    let a = t.get_or_insert_query_with(rt, key(0, 0, 0), reads(1), type_fallback());
    let b = t.insert_queries_watched(rt, vec![(key(0, 1, 0), reads(0), type_fallback())]);
    assert_eq!(rt.get(a), 7);
    assert_eq!(rt.get(b[0]), 7);
}

fn grid_a_lazy_query_is_built_only_on_a_miss<C: Cells>(rt: &Runtime<C>)
where
    GridTable<i64, C>: Send + Sync,
{
    let t = Arc::new(rt.create_grid_table::<i64>());
    let builds = Arc::new(AtomicU32::new(0));
    let (t2, b2) = (t.clone(), builds.clone());
    let cyclic = t.get_or_insert_query_lazy(rt, key(0, 0, 0), move || {
        b2.fetch_add(1, Ordering::Relaxed);
        let t3 = t2.clone();
        let reads_itself = move |rt: &Runtime<C>| {
            let me = t3.get_or_insert_query_lazy(rt, key(0, 0, 0), || -> Built<C> {
                unreachable!("a hit inside a compute builds nothing")
            });
            rt.get(me) + 1
        };
        let opts = QueryOpts {
            cycle: CycleStrategy::fallback(|_| -5),
            ..Default::default()
        };
        (reads_itself, opts)
    });
    assert_eq!(builds.load(Ordering::Relaxed), 1);
    assert_eq!(rt.get(cyclic), -5, "the options built on the miss apply");
    let again = t.get_or_insert_query_lazy(rt, key(0, 0, 0), || -> Built<C> {
        unreachable!("a hit builds nothing")
    });
    assert_eq!(again, cyclic);

    let (t2, b2) = (t.clone(), builds.clone());
    let outer = rt.create_query(move |rt| {
        let b3 = b2.clone();
        let inner = t2.get_or_insert_query_lazy(rt, key(0, 1, 0), move || -> Built<C> {
            b3.fetch_add(1, Ordering::Relaxed);
            (|_| 4, QueryOpts::default())
        });
        rt.get(inner) * 10
    });
    assert_eq!(rt.get(outer), 40);
    assert_eq!(
        builds.load(Ordering::Relaxed),
        2,
        "a miss inside a compute builds once"
    );
    assert!(t.get(rt, key(0, 1, 0)).is_some());
}

#[cfg(all(debug_assertions, not(loom)))]
mod lock_order {
    use super::*;
    use crate::persist_hooks::{get_or_insert_query_seeded, Deferred};

    both!(grid_a_miss_under_a_table_miss_breaks_the_lock_order);

    /// A seed hook runs under its `NodeTable`'s miss mutex, so a grid
    /// miss there would hold two table misses at once.
    fn grid_a_miss_under_a_table_miss_breaks_the_lock_order<C: Cells>(rt: &Runtime<C>) {
        let grid = rt.create_grid_table::<i64>();
        let table = rt.create_table::<u8, i64>();
        let mut bag = Deferred::new();
        panics_with("taking TableMiss while holding TableMiss", || {
            get_or_insert_query_seeded(
                &table,
                rt,
                0,
                |_: &Runtime<C>| 1,
                Default::default(),
                |rt, _| {
                    grid.get_or_insert_input(rt, key(0, 0, 0), || 2);
                    None
                },
                &mut bag,
            );
        });
        bag.drain(rt);
    }
}
