//! T8's operator rules: a cursor-driven operator that runs as a cycle
//! member, or that read a provisional value, lands the output a fresh
//! operator built after the episode would have, and a `group_by` router
//! refuses a provisional value with the cycle and routes nothing.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, OnceLock};

use super::*;
use crate::cells::{Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts};
use crate::state::NodeState;

/// A runtime a user function captures to read nodes, which the operator
/// API's `Fn(&T)` signatures otherwise can't reach.
struct RtRef<C: Cells>(*const Runtime<C>);

// SAFETY: each test keeps its `OwnedRt` alive past every operator run, and
// a `Local` runtime's operators only run on the test's thread.
unsafe impl<C: Cells> Send for RtRef<C> {}
// SAFETY: as above.
unsafe impl<C: Cells> Sync for RtRef<C> {}

impl<C: Cells> RtRef<C> {
    fn rt(&self) -> &Runtime<C> {
        // SAFETY: as above.
        unsafe { &*self.0 }
    }
}

/// A runtime owned through a raw pointer. Moving a `Box` retags its
/// allocation as unique, which under Stacked Borrows invalidates every
/// `RtRef` taken before the move; moving a raw pointer retags nothing.
struct OwnedRt<C: Cells>(NonNull<Runtime<C>>);

impl<C: Cells> OwnedRt<C> {
    fn new() -> Self {
        Self(NonNull::new(Box::into_raw(Box::new(Runtime::new()))).unwrap())
    }

    fn r(&self) -> RtRef<C> {
        RtRef(self.0.as_ptr())
    }
}

impl<C: Cells> std::ops::Deref for OwnedRt<C> {
    type Target = Runtime<C>;

    fn deref(&self) -> &Runtime<C> {
        // SAFETY: the pointer came from `Box::into_raw` and is freed only by
        // `drop`, so it is valid for as long as `self` is.
        unsafe { self.0.as_ref() }
    }
}

impl<C: Cells> Drop for OwnedRt<C> {
    fn drop(&mut self) {
        // SAFETY: as in `deref`; nothing reads through an `RtRef` after the
        // test drops its `OwnedRt`.
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

fn fixpoint(limit: u32) -> QueryOpts<u64> {
    QueryOpts {
        cycle: CycleStrategy::Fixpoint(Fixpoint::new(|| 0_u64).max_iterations(limit)),
        ..QueryOpts::default()
    }
}

fn sorted(mut v: Vec<u64>) -> Vec<u64> {
    v.sort_unstable();
    v
}

/// M = min(count(F) + 1, 5) and F = S.filter(x <= M), over S = 0..10, so
/// M climbs 0, 2, 4, 5 and F lands {0..=5}.
fn filter_member<C: Cells>() {
    let rt = OwnedRt::<C>::new();
    let s = rt.create_collection::<u64>();
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let (r, m) = (rt.r(), m_cell.clone());
    let f = s.filter(&rt, move |&x| x <= r.rt().get(*m.get().unwrap()));
    let count = f.count(&rt);
    let m = rt.create_query_with(move |rt| (rt.get(count) + 1).min(5), fixpoint(20));
    m_cell.set(m).unwrap();
    for x in 0..10 {
        s.insert(&rt, x);
    }

    assert_eq!(rt.get(m), 5);
    let landed = rt.get(m);
    let fresh = s.filter(&rt, move |&x| x <= landed);
    assert_eq!(sorted(f.values(&rt)), sorted(fresh.values(&rt)));
    assert_eq!(rt.get(count), 6);
    rt.assert_invariant_i();
}

/// As [`filter_member`], with an aggregate counting the members at or
/// below M in place of the filter and its count.
fn aggregate_member<C: Cells>() {
    let rt = OwnedRt::<C>::new();
    let s = rt.create_collection::<u64>();
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let (r, m) = (rt.r(), m_cell.clone());
    let below = s.aggregate(
        &rt,
        0_u64,
        move |&x| u64::from(x <= r.rt().get(*m.get().unwrap())),
        |a, b| a + b,
    );
    let m = rt.create_query_with(move |rt| (rt.get(below) + 1).min(5), fixpoint(20));
    m_cell.set(m).unwrap();
    for x in 0..10 {
        s.insert(&rt, x);
    }

    assert_eq!(rt.get(m), 5);
    let landed = rt.get(m);
    let fresh = s.aggregate(&rt, 0_u64, move |&x| u64::from(x <= landed), |a, b| a + b);
    assert_eq!(rt.get(below), rt.get(fresh));
    assert_eq!(rt.get(below), 6);
    rt.assert_invariant_i();
}

/// As [`aggregate_member`], with `combine` reading M in place of `lift`:
/// `min(a + b, M + 1)` is associative for nonnegative values, so the
/// aggregate counts `min(|S|, M + 1)`.
fn aggregate_combine_member<C: Cells>() {
    let rt = OwnedRt::<C>::new();
    let s = rt.create_collection::<u64>();
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let (r, m) = (rt.r(), m_cell.clone());
    let below = s.aggregate(
        &rt,
        0_u64,
        |_| 1,
        move |a, b| (a + b).min(r.rt().get(*m.get().unwrap()) + 1),
    );
    let m = rt.create_query_with(move |rt| (rt.get(below) + 1).min(5), fixpoint(20));
    m_cell.set(m).unwrap();
    for x in 0..10 {
        s.insert(&rt, x);
    }

    assert_eq!(rt.get(m), 5);
    let landed = rt.get(m);
    let fresh = s.aggregate(&rt, 0_u64, |_| 1, move |a, b| (a + b).min(landed + 1));
    assert_eq!(rt.get(below), rt.get(fresh));
    assert_eq!(rt.get(below), 6);
    rt.assert_invariant_i();
}

/// A router whose `key_fn` reads Fixpoint member M while M reads the
/// router: the router's run gets `Err(Cycle)` at M's `try_get` and leaves
/// its groups, their version nodes and its key cache as they were. Once a
/// `set` unlinks M from `key_fn`, it routes normally. Element 0 is routed
/// at construction, so the state the refusals leave alone is not empty.
fn group_by_refuses<C: Cells>() {
    let rt = OwnedRt::<C>::new();
    let s = rt.create_collection::<u64>();
    s.insert(&rt, 0);
    let link = rt.create_input(1_u64);
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let (r, m) = (rt.r(), m_cell.clone());
    let grouped = s.group_by(&rt, move |&x| {
        let rt = r.rt();
        if x == 0 {
            0
        } else if rt.get(link) == 1 {
            (x + rt.get(*m.get().unwrap())) % 2
        } else {
            x % 2
        }
    });
    let router = grouped.version_node();
    let refused = Arc::new(AtomicBool::new(false));
    let seen = refused.clone();
    let m = rt.create_query_with(
        move |rt| match rt.try_get(router) {
            Ok(v) => v,
            Err(_) => {
                seen.store(true, SeqCst);
                7
            }
        },
        fixpoint(20),
    );
    m_cell.set(m).unwrap();
    for x in 1..=3 {
        s.insert(&rt, x);
    }

    let routed = grouped.routed_state(&rt);
    assert_eq!(grouped.group_count(), 1);
    let _ = rt.get(m);
    assert!(refused.load(SeqCst), "the router read a provisional M");
    assert_eq!(
        grouped.routed_state(&rt),
        routed,
        "a refused run routes nothing"
    );

    // The refusal left the router marked for a rebuild, whose key_fn pass
    // over the snapshot meets M again. The read's walk runs the router as
    // M's dep, before M's closure, so the cycle reaches this `try_get`.
    s.insert(&rt, 4);
    assert!(rt.try_get(m).is_err(), "the rebuild read a provisional M");
    assert_eq!(
        grouped.routed_state(&rt),
        routed,
        "a refused rebuild routes nothing"
    );

    rt.set(link, 0);
    let _ = rt.get(m);
    let mut keys = grouped.keys();
    keys.sort_unstable();
    assert_eq!(keys, vec![0, 1]);
    assert_eq!(
        sorted(grouped.get_group(&1).unwrap().values(&rt)),
        vec![1, 3]
    );
    rt.assert_invariant_i();
}

/// The provisional value reaches the router through Q, a query that is no
/// member: Q reads M on the stack, defers, and hands its frame `LOW`.
/// Element 0 is routed at construction, as in [`group_by_refuses`].
fn group_by_refuses_through_a_deferral<C: Cells>() {
    let rt = OwnedRt::<C>::new();
    let s = rt.create_collection::<u64>();
    s.insert(&rt, 0);
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let m = m_cell.clone();
    let q = rt.create_query(move |rt| rt.get(*m.get().unwrap()) % 2);
    let r = rt.r();
    let grouped = s.group_by(
        &rt,
        move |&x| {
            if x == 0 {
                0
            } else {
                (x + r.rt().get(q)) % 2
            }
        },
    );
    let router = grouped.version_node();
    let m = rt.create_query_with(move |rt| rt.try_get(router).unwrap_or(7), fixpoint(20));
    m_cell.set(m).unwrap();
    for x in 1..=3 {
        s.insert(&rt, x);
    }
    let routed = grouped.routed_state(&rt);
    assert_eq!(grouped.group_count(), 1);
    assert_eq!(rt.get(m), 7, "the router refused");
    assert_eq!(grouped.routed_state(&rt), routed);
    rt.assert_invariant_i();
}

/// The router's upstream is a filter whose `pred` reads M, and the episode
/// is entered at M, so the filter is a member off the stack when the
/// router reads its version.
fn group_by_refuses_a_member_upstream<C: Cells>() {
    let rt = OwnedRt::<C>::new();
    let s = rt.create_collection::<u64>();
    let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let (r, m) = (rt.r(), m_cell.clone());
    let f = s.filter(&rt, move |&x| x <= r.rt().get(*m.get().unwrap()));
    let grouped = f.group_by(&rt, |&x| x % 2);
    let router = grouped.version_node();
    let m = rt.create_query_with(move |rt| rt.try_get(router).unwrap_or(7), fixpoint(20));
    m_cell.set(m).unwrap();
    for x in 1..=3 {
        s.insert(&rt, x);
    }
    let routed = grouped.routed_state(&rt);
    assert_eq!(rt.get(m), 7, "the router refused");
    assert_eq!(grouped.routed_state(&rt), routed);
    assert_eq!(grouped.group_count(), 0);
    rt.assert_invariant_i();
}

/// Called with the number of the run it is called from.
type Hook = Arc<dyn Fn(usize) + Send + Sync>;

/// [`filter_member`]'s shape, M = min(count(F) + 1, 5) + (W on one run)
/// and F = S.filter(pred) over S = 0..10, where M reads `count` with
/// `try_get`, so that an entry below M meets R3 as an `Err`, with ways to change F's reads
/// part-way through the episode. `pred` is `x <= M` for its first
/// `m_reads` calls while input `link` is 1, and `x <= 3` after that.
#[derive(Clone, Default)]
struct Shape {
    m_reads: usize,
    link: u64,
    /// M's run that sets `link` to 0.
    unlink_on: Option<usize>,
    /// M's run that reads W, which reads `count` and lands 0.
    w_on: Option<usize>,
    m_hook: Option<Hook>,
    w_hook: Option<Hook>,
}

impl Shape {
    fn plain() -> Self {
        Self {
            m_reads: usize::MAX,
            link: 1,
            ..Self::default()
        }
    }

    fn unlinked() -> Self {
        Self {
            link: 0,
            ..Self::plain()
        }
    }
}

struct Episode<C: Cells> {
    rt: OwnedRt<C>,
    s: IncrCollection<u64, C>,
    f: IncrCollection<u64, C>,
    count: Incr<u64>,
    m: Incr<u64>,
    w: Incr<u64>,
    /// Runs of `pred`, M and W.
    runs: Arc<AtomicUsize>,
}

impl<C: Cells> Episode<C> {
    fn new(shape: Shape) -> Self {
        let rt = OwnedRt::<C>::new();
        let s = rt.create_collection::<u64>();
        let link = rt.create_input(shape.link);
        let runs = Arc::new(AtomicUsize::new(0));
        let m_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
        let pred_calls = AtomicUsize::new(0);
        let (r, m, pred_runs) = (rt.r(), m_cell.clone(), runs.clone());
        let f = s.filter(&rt, move |&x| {
            pred_runs.fetch_add(1, SeqCst);
            let rt = r.rt();
            if pred_calls.fetch_add(1, SeqCst) < shape.m_reads && rt.get(link) == 1 {
                x <= rt.get(*m.get().unwrap())
            } else {
                x <= 3
            }
        });
        let count = f.count(&rt);
        let (w_runs, w_hook) = (runs.clone(), shape.w_hook.clone());
        let w = rt.create_query(move |rt| {
            w_runs.fetch_add(1, SeqCst);
            if let Some(hook) = &w_hook {
                hook(1);
            }
            rt.get(count);
            0
        });
        let (m_runs, m_hook) = (runs.clone(), shape.m_hook.clone());
        let own_runs = AtomicUsize::new(0);
        let m = rt.create_query_with(
            move |rt| {
                m_runs.fetch_add(1, SeqCst);
                let run = own_runs.fetch_add(1, SeqCst) + 1;
                if let Some(hook) = &m_hook {
                    hook(run);
                }
                let w = if shape.w_on == Some(run) {
                    rt.get(w)
                } else {
                    0
                };
                if shape.unlink_on == Some(run) {
                    rt.set(link, 0);
                }
                (rt.try_get(count).unwrap_or(0) + 1).min(5) + w
            },
            fixpoint(20),
        );
        m_cell.set(m).unwrap();
        for x in 0..10 {
            s.insert(&rt, x);
        }
        Self {
            rt,
            s,
            f,
            count,
            m,
            w,
            runs,
        }
    }

    /// The episode's landing against a fresh runtime of shape `fresh` and
    /// a fresh filter `x <= bound`: equal values, every member `Clean`,
    /// and a further get of each node runs no closure. A run that spans a
    /// `set` may land `Dirty` and run again, as in
    /// `set_during_replay_dissolves`, so with `spans_set` that holds only
    /// from the get after it.
    fn assert_landed(&self, fresh: Shape, bound: u64, spans_set: bool) {
        let rt = &*self.rt;
        let version = self.f.version_node();
        let assert_clean = || {
            for (name, state) in [
                ("M", rt.state_of(self.m)),
                ("count", rt.state_of(self.count)),
                ("F", rt.state_of(version)),
            ] {
                assert_eq!(state, NodeState::Clean, "{name} landed Clean");
            }
        };
        if spans_set {
            let _ = (rt.get(self.m), rt.get(self.count), rt.get(version));
        }
        assert_clean();
        let runs = self.runs.load(SeqCst);
        let landed = (rt.get(self.m), rt.get(self.count), rt.get(version));
        assert_eq!(self.runs.load(SeqCst), runs, "a further get ran a closure");
        assert_clean();
        let landed = (landed.0, landed.1);

        let other = Episode::<C>::new(fresh);
        assert_eq!(landed, (other.rt.get(other.m), other.rt.get(other.count)));
        let fresh_op = self.s.filter(rt, move |&x| x <= bound);
        assert_eq!(sorted(self.f.values(rt)), sorted(fresh_op.values(rt)));
        assert_eq!(rt.cycle_of(self.w), None);
        rt.assert_invariant_i();
    }
}

/// `pred` reads M only in the discovery run, so the episode's validation
/// releases F and the cycle dissolves: F lands `x <= 3` as a fresh filter
/// does, over a rebuild rather than its provisional output.
fn dissolve_after_discovery<C: Cells>() {
    let ep = Episode::<C>::new(Shape {
        m_reads: 10,
        ..Shape::plain()
    });
    assert_eq!(ep.rt.get(ep.m), 5);
    assert_eq!(ep.rt.cycle_of(ep.m), None);
    ep.assert_landed(Shape::unlinked(), 3, false);
}

/// M's iteration-1 run sets `link`, which takes M out of `pred`'s reads,
/// as `set_during_replay_dissolves`' Z does.
fn dissolve_after_iteration_1<C: Cells>() {
    let ep = Episode::<C>::new(Shape {
        unlink_on: Some(2),
        ..Shape::plain()
    });
    assert_eq!(ep.rt.get(ep.m), 5);
    assert_eq!(ep.rt.cycle_of(ep.m), None);
    ep.assert_landed(Shape::unlinked(), 3, true);
}

/// M reads W, which reads `count`, only on one run. M's last run doesn't
/// read W, so W is on no cycle once the episode lands. Read on M's run 1 or
/// 2, validation releases it; read first in iteration 2 (run 3) or later,
/// it joins after validation and the landing's cut releases it.
fn member_read_once_leaves_the_group<C: Cells>() {
    for on in 1..=4 {
        let ep = Episode::<C>::new(Shape {
            w_on: Some(on),
            ..Shape::plain()
        });
        assert_eq!(ep.rt.get(ep.m), 5);
        assert_eq!(ep.rt.cycle_of(ep.w), None, "W read on M's run {on}");
        ep.assert_landed(Shape::plain(), 5, false);
    }
}

#[test]
fn local_member_read_once_leaves_the_group() {
    member_read_once_leaves_the_group::<Local>();
}

#[test]
fn shared_member_read_once_leaves_the_group() {
    member_read_once_leaves_the_group::<Shared>();
}

#[test]
fn local_filter_dissolves_after_the_discovery_run() {
    dissolve_after_discovery::<Local>();
}

#[test]
fn shared_filter_dissolves_after_the_discovery_run() {
    dissolve_after_discovery::<Shared>();
}

#[test]
fn local_filter_dissolves_after_iteration_1() {
    dissolve_after_iteration_1::<Local>();
}

#[test]
fn shared_filter_dissolves_after_iteration_1() {
    dissolve_after_iteration_1::<Shared>();
}

/// Thread 1 runs the episode at M. On M's run `on`, M parks before it
/// reads W, which thread 2 has claimed and parks in before W reads
/// `count`, a member thread 1 holds. Each then waits on the other, the
/// episode's thread first unless `episode_older` is false, and the
/// registry wounds the younger.
#[cfg(not(loom))]
mod wounds {
    use std::thread;

    use super::*;
    use crate::runtime::race_tests::{release_parked_on_unwind, Gate};
    use crate::runtime::wound_tests::{assert_wounded, wait_registered};

    fn wound_at(on: usize, episode_older: bool) {
        let (m_gate, w_gate) = (Gate::new(), Gate::new());
        let (mg, wg) = (m_gate.clone(), w_gate.clone());
        let ep = Episode::<Shared>::new(Shape {
            w_on: Some(on),
            m_hook: Some(Arc::new(move |run| {
                if run == on {
                    mg.park();
                }
            })),
            w_hook: Some(Arc::new(move |_| wg.park())),
            ..Shape::plain()
        });
        let rt = &*ep.rt;
        let (m, w) = (ep.m, ep.w);
        thread::scope(|scope| {
            let _released = release_parked_on_unwind();
            m_gate.arm();
            let episode = scope.spawn(move || rt.get(m));
            m_gate.wait_parked();
            w_gate.arm();
            let reader = scope.spawn(move || rt.get(w));
            w_gate.wait_parked();
            let (first, second) = if episode_older {
                (&m_gate, &w_gate)
            } else {
                (&w_gate, &m_gate)
            };
            first.release();
            wait_registered(rt.runtime_id, 1);
            second.release();
            assert_eq!(episode.join().unwrap(), 5);
            assert_eq!(reader.join().unwrap(), 0);
        });
        assert_wounded();
        ep.assert_landed(Shape::plain(), 5, false);
    }

    macro_rules! wound_tests {
        ($($(#[$attr:meta])* $name:ident: $on:expr, $older:expr;)*) => {$(
            #[test]
            #[cfg_attr(miri, ignore)]
            $(#[$attr])*
            fn $name() {
                crate::child_test::run_in_child(
                    concat!("collection::cycle_tests::wounds::", stringify!($name)),
                    || wound_at($on, $older),
                )
                .assert_clean();
            }
        )*};
    }

    wound_tests! {
        shared_wound_after_the_discovery_run_episode_older: 2, true;
        shared_wound_after_the_discovery_run_reader_older: 2, false;
        shared_wound_after_iteration_1_episode_older: 3, true;
        shared_wound_after_iteration_1_reader_older: 3, false;
    }
}

#[test]
fn local_group_by_refuses_through_a_deferral() {
    group_by_refuses_through_a_deferral::<Local>();
}

#[test]
fn shared_group_by_refuses_through_a_deferral() {
    group_by_refuses_through_a_deferral::<Shared>();
}

#[test]
fn local_group_by_refuses_a_member_upstream() {
    group_by_refuses_a_member_upstream::<Local>();
}

#[test]
fn shared_group_by_refuses_a_member_upstream() {
    group_by_refuses_a_member_upstream::<Shared>();
}

#[test]
fn local_filter_member_lands_a_fresh_filters_output() {
    filter_member::<Local>();
}

#[test]
fn shared_filter_member_lands_a_fresh_filters_output() {
    filter_member::<Shared>();
}

#[test]
fn local_aggregate_member_lands_a_fresh_aggregates_output() {
    aggregate_member::<Local>();
}

#[test]
fn shared_aggregate_member_lands_a_fresh_aggregates_output() {
    aggregate_member::<Shared>();
}

#[test]
fn local_aggregate_combine_member_lands_a_fresh_aggregates_output() {
    aggregate_combine_member::<Local>();
}

#[test]
fn shared_aggregate_combine_member_lands_a_fresh_aggregates_output() {
    aggregate_combine_member::<Shared>();
}

#[test]
fn local_group_by_refuses_a_provisional_key() {
    group_by_refuses::<Local>();
}

#[test]
fn shared_group_by_refuses_a_provisional_key() {
    group_by_refuses::<Shared>();
}
