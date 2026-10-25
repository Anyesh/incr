//! The restore and checkpoint hooks (T10): `seed_base`, `seed_query`,
//! `NodeTable::get_or_insert_query_seeded` and `inspect`.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::SeqCst};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use super::race_tests::{id, park_at, Gate};
use super::{ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::child_test::run_in_child;
use crate::handle::{ErasedIncr, Incr};
use crate::persist_hooks::{
    changed_at, changed_at_erased, frame_doomed, get_or_insert_query_seeded, inspect,
    pending_handle, probe_unchecked, seed_base, seed_query, Deferred, Reuse, SeedDep, SeedError,
};
use crate::reclaim_probe::{self, RetireSite};
use crate::state::NodeState;
use crate::table::NodeTable;

type Rt = Runtime<Shared>;

macro_rules! both {
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

both!(
    if_unchanged_over_unchanged_deps_never_runs,
    recompute_first_runs_once_and_cuts_off,
    a_recompute_first_seed_runs_no_dep_its_rerun_drops,
    a_moved_trusted_dep_downgrades_to_recompute_first,
    failures_return_the_value_and_land_new,
    two_restores_in_one_compute_run_once,
    recomputed_seeded_dep_invalidates_trusting_entry,
    inspect_reads_a_clean_node_with_its_deps,
    table_seeds_only_a_node_it_creates,
    a_panicking_seed_hook_leaves_no_node,
    changed_at_erased_refuses_dead_and_foreign_handles,
    a_doomed_frame_sees_frame_doomed_and_reads_the_pending_handle,
    a_seeded_hit_leaves_its_key_and_closure_in_the_bag,
);

fn dep<T>(h: Incr<T>) -> SeedDep {
    SeedDep {
        dep: h.erase(),
        trusted_stamp: None,
    }
}

fn trusted<T>(h: Incr<T>, stamp: u64) -> SeedDep {
    SeedDep {
        dep: h.erase(),
        trusted_stamp: Some(stamp),
    }
}

/// A query over `x` that counts its runs and returns `x * 10`.
fn counted<C: Cells>(rt: &Runtime<C>, x: Incr<u64>) -> (Incr<u64>, Arc<AtomicU32>) {
    let runs = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&runs);
    let q = rt.create_query(move |rt: &Runtime<C>| {
        counter.fetch_add(1, SeqCst);
        rt.get(x) * 10
    });
    (q, runs)
}

/// Seeded `IfUnchanged` over an input that has not changed since the
/// base: the node lands Dirty at W0 and verifies Clean without running,
/// and a later change runs it.
fn if_unchanged_over_unchanged_deps_never_runs<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(3_u64);
    let (q, runs) = counted(&rt, x);
    let base = seed_base(&rt);
    assert_eq!(
        seed_query(&rt, &base, q, 30, &[dep(x)], Reuse::IfUnchanged),
        Ok(Reuse::IfUnchanged)
    );
    assert_eq!(rt.state_of(q), NodeState::Dirty);
    let node = rt.nodes.get(q.slot());
    assert_eq!((node.changed_at(), node.verified_at()), (base.w0, base.w0));
    assert_eq!(base.w0 % 2, 1, "W0 is an odd stamp");
    assert_eq!(rt.get(q), 30);
    assert_eq!(runs.load(SeqCst), 0);
    assert_eq!(rt.state_of(q), NodeState::Clean);

    rt.set(x, 4);
    assert_eq!(rt.get(q), 40);
    assert_eq!(runs.load(SeqCst), 1);
    rt.assert_invariant_i();
}

/// `RecomputeFirst` runs the closure on the first read; a run that
/// returns the seeded value is a cutoff, so a dependent seeded over it
/// stays verified.
fn recompute_first_runs_once_and_cuts_off<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(3_u64);
    let (q, runs) = counted(&rt, x);
    let (r, r_runs) = {
        let r_runs = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&r_runs);
        let r = rt.create_query(move |rt: &Runtime<C>| {
            counter.fetch_add(1, SeqCst);
            rt.get(q) + 1
        });
        (r, r_runs)
    };
    let base = seed_base(&rt);
    assert_eq!(
        seed_query(&rt, &base, q, 30, &[dep(x)], Reuse::RecomputeFirst),
        Ok(Reuse::RecomputeFirst)
    );
    assert_eq!(rt.nodes.get(q.slot()).verified_at(), 0);
    assert_eq!(
        seed_query(
            &rt,
            &base,
            r,
            31,
            &[trusted(q, base.w0)],
            Reuse::IfUnchanged
        ),
        Ok(Reuse::IfUnchanged)
    );
    assert_eq!(rt.get(r), 31);
    assert_eq!(runs.load(SeqCst), 1);
    assert_eq!(r_runs.load(SeqCst), 0, "a cutoff recompute of q ran r");
    assert_eq!(rt.nodes.get(q.slot()).changed_at(), base.w0);
    rt.assert_invariant_i();
}

/// Seeded `RecomputeFirst` over a stale run that read Dirty `a`: the rerun
/// reads only `p`, so `a` never runs, as in a cold run.
fn a_recompute_first_seed_runs_no_dep_its_rerun_drops<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let (a, a_runs) = counted(&rt, x);
    assert_eq!(rt.get(a), 10);
    rt.set(x, 2);
    let p = rt.create_input(5_u64);
    let q = rt.create_query(move |rt: &Runtime<C>| rt.get(p));
    let base = seed_base(&rt);
    assert_eq!(
        seed_query(&rt, &base, q, 10, &[dep(p), dep(a)], Reuse::RecomputeFirst),
        Ok(Reuse::RecomputeFirst)
    );
    assert_eq!(rt.get(q), 5);
    assert_eq!(
        a_runs.load(SeqCst),
        1,
        "the walk ran a dep q no longer reads"
    );
    rt.assert_invariant_i();
    rt.assert_edges_complete();
}

/// A trusted dep whose stamp is not the one recorded, or that is not
/// Clean or Dirty, makes the seed `RecomputeFirst`.
fn a_moved_trusted_dep_downgrades_to_recompute_first<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(3_u64);
    let (q, _) = counted(&rt, x);
    assert_eq!(rt.get(q), 30);
    let (r, _) = counted(&rt, x);
    let base = seed_base(&rt);
    let stamp = rt.nodes.get(q.slot()).changed_at();
    assert_eq!(
        seed_query(
            &rt,
            &base,
            r,
            0,
            &[trusted(q, stamp + 2), dep(x)],
            Reuse::IfUnchanged
        ),
        Ok(Reuse::RecomputeFirst)
    );
    assert_eq!(rt.nodes.get(r.slot()).verified_at(), 0);
    assert_eq!(rt.get(r), 30);

    let failed = rt.create_query(|_: &Runtime<C>| -> u64 { panic!("boom") });
    assert!(catch_unwind(AssertUnwindSafe(|| rt.get(failed))).is_err());
    assert_eq!(rt.state_of(failed), NodeState::Failed);
    let (s, _) = counted(&rt, x);
    let stamp = rt.nodes.get(failed.slot()).changed_at();
    assert_eq!(
        seed_query(
            &rt,
            &seed_base(&rt),
            s,
            0,
            &[trusted(failed, stamp)],
            Reuse::IfUnchanged
        ),
        Ok(Reuse::RecomputeFirst)
    );
}

/// Each failure hands the value back, lands the node New and installs
/// nothing, so a read computes it cold.
fn failures_return_the_value_and_land_new<C: Cells>() {
    let rt = Runtime::<C>::new();
    let other = Runtime::<C>::new();
    let far = (0..64_u64).map(|i| other.create_input(i)).last().unwrap();
    let x = rt.create_input(3_u64);
    let gone = rt.create_input(0_u64);
    rt.delete_node(gone);
    let (q, runs) = counted(&rt, x);
    let base = seed_base(&rt);
    let foreign = seed_base(&other);
    let cases: [(
        &crate::persist_hooks::SeedBase,
        Vec<SeedDep>,
        Reuse,
        SeedError,
    ); 4] = [
        (
            &foreign,
            vec![dep(x)],
            Reuse::IfUnchanged,
            SeedError::ForeignBase,
        ),
        (
            &base,
            vec![dep(gone)],
            Reuse::IfUnchanged,
            SeedError::DepNotAlive,
        ),
        (
            &base,
            vec![dep(far)],
            Reuse::IfUnchanged,
            SeedError::DepNotAlive,
        ),
        (&base, vec![], Reuse::RecomputeFirst, SeedError::NoDeps),
    ];
    for (b, deps, reuse, err) in cases {
        assert_eq!(seed_query(&rt, b, q, 7, &deps, reuse), Err((err, 7)));
        assert_eq!(rt.state_of(q), NodeState::New);
        assert_eq!(rt.nodes.get(q.slot()).dep_count(), 0);
    }
    assert_eq!(rt.get(q), 30);
    assert_eq!(runs.load(SeqCst), 1);
    assert_eq!(
        seed_query(&rt, &base, q, 7, &[dep(x)], Reuse::IfUnchanged),
        Err((SeedError::NotFresh, 7))
    );
    assert_eq!(rt.state_of(q), NodeState::Clean);
    rt.assert_invariant_i();
}

/// P restores A over x and B over A from a table inside its compute and
/// reads both: A and B never run and P runs once.
fn two_restores_in_one_compute_run_once<C: Cells>()
where
    NodeTable<u8, u64, C>: Send + Sync,
{
    let rt = Runtime::<C>::new();
    let x = rt.create_input(2_u64);
    let table: Arc<NodeTable<u8, u64, C>> = Arc::new(rt.create_table());
    let leaf_runs = Arc::new(AtomicU32::new(0));
    let p_runs = Arc::new(AtomicU32::new(0));
    let p = {
        let (table, leaf_runs, p_runs) = (table.clone(), leaf_runs.clone(), p_runs.clone());
        rt.create_query(move |rt: &Runtime<C>| {
            p_runs.fetch_add(1, SeqCst);
            let base = seed_base(rt);
            let mut bag = Deferred::new();
            let runs = leaf_runs.clone();
            let a = get_or_insert_query_seeded(
                &table,
                rt,
                0,
                move |rt: &Runtime<C>| {
                    runs.fetch_add(1, SeqCst);
                    rt.get(x) * 10
                },
                Default::default(),
                |rt, a| {
                    seed_query(rt, &base, a, 20, &[dep(x)], Reuse::IfUnchanged)
                        .err()
                        .map(|e| e.1)
                },
                &mut bag,
            );
            let runs = leaf_runs.clone();
            let b = get_or_insert_query_seeded(
                &table,
                rt,
                1,
                move |rt: &Runtime<C>| {
                    runs.fetch_add(1, SeqCst);
                    rt.get(a) + 1
                },
                Default::default(),
                |rt, b| {
                    seed_query(rt, &base, b, 21, &[trusted(a, base.w0)], Reuse::IfUnchanged)
                        .err()
                        .map(|e| e.1)
                },
                &mut bag,
            );
            bag.drain(rt);
            rt.get(a) + rt.get(b)
        })
    };
    assert_eq!(rt.get(p), 41);
    assert_eq!(p_runs.load(SeqCst), 1);
    assert_eq!(leaf_runs.load(SeqCst), 0);
    rt.assert_invariant_i();
}

/// R trusts A at its seed stamp; A recomputes to a new value, which
/// stamps above W0, so R runs.
fn recomputed_seeded_dep_invalidates_trusting_entry<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(5_u64);
    let (a, _) = counted(&rt, x);
    let (r, r_runs) = {
        let r_runs = Arc::new(AtomicU32::new(0));
        let counter = Arc::clone(&r_runs);
        let r = rt.create_query(move |rt: &Runtime<C>| {
            counter.fetch_add(1, SeqCst);
            rt.get(a) + 1
        });
        (r, r_runs)
    };
    let base = seed_base(&rt);
    assert_eq!(
        seed_query(&rt, &base, a, 999, &[dep(x)], Reuse::RecomputeFirst),
        Ok(Reuse::RecomputeFirst)
    );
    assert_eq!(
        seed_query(
            &rt,
            &base,
            r,
            1000,
            &[trusted(a, base.w0)],
            Reuse::IfUnchanged
        ),
        Ok(Reuse::IfUnchanged)
    );
    assert_eq!(rt.get(r), 51);
    assert_eq!(r_runs.load(SeqCst), 1);
    assert!(rt.nodes.get(a.slot()).changed_at() > base.w0);
    rt.assert_invariant_i();
}

/// `inspect` returns a Clean node's value and deps, and `None` for one
/// that is not Clean.
fn inspect_reads_a_clean_node_with_its_deps<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let y = rt.create_input(2_u64);
    let q = rt.create_query(move |rt: &Runtime<C>| rt.get(x) + rt.get(y));
    assert!(inspect(&rt, q).is_none());
    assert_eq!(rt.get(q), 3);
    let seen = inspect(&rt, q).expect("a Clean node is inspected");
    assert_eq!(seen.value, 3);
    assert_eq!(seen.deps, vec![x.erase(), y.erase()]);
    assert!(!seen.in_group);
    rt.set(x, 5);
    assert!(inspect(&rt, q).is_none());
}

/// The seed hook runs only for a node the miss creates; a hit, or the
/// probe, returns the live node and leaves the hook uncalled.
fn table_seeds_only_a_node_it_creates<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(4_u64);
    let table: NodeTable<u8, u64, C> = rt.create_table();
    assert_eq!(probe_unchecked(&table, &rt, &0), None);
    let calls = AtomicU32::new(0);
    let hook = |rt: &Runtime<C>, q: Incr<u64>| {
        calls.fetch_add(1, SeqCst);
        let base = seed_base(rt);
        seed_query(rt, &base, q, 40, &[dep(x)], Reuse::IfUnchanged)
            .err()
            .map(|e| e.1)
    };
    let f = move |rt: &Runtime<C>| rt.get(x) * 10;
    let mut bag = Deferred::new();
    let q = get_or_insert_query_seeded(&table, &rt, 0, f, Default::default(), hook, &mut bag);
    assert_eq!(rt.state_of(q), NodeState::Dirty);
    let again = get_or_insert_query_seeded(
        &table,
        &rt,
        0,
        f,
        Default::default(),
        |_, _| panic!("the hook ran on a hit"),
        &mut bag,
    );
    bag.drain(&rt);
    assert_eq!(again, q);
    assert_eq!(calls.load(SeqCst), 1);
    assert_eq!(probe_unchecked(&table, &rt, &0), Some(q));
    assert_eq!(rt.get(q), 40);
}

/// A seed hook that panics, before seeding or after seeding with a dep,
/// leaves neither its node nor a table entry, and the next miss creates
/// the key's node.
fn a_panicking_seed_hook_leaves_no_node<C: Cells>() {
    for seed_first in [false, true] {
        let rt = Runtime::<C>::new();
        let x = rt.create_input(4_u64);
        let table: NodeTable<u8, u64, C> = rt.create_table();
        let nodes = rt.node_count();
        let created = Cell::new(None);
        let hook = |rt: &Runtime<C>, q: Incr<u64>| -> Option<u64> {
            created.set(Some(q));
            if seed_first {
                let base = seed_base(rt);
                seed_query(rt, &base, q, 40, &[dep(x)], Reuse::IfUnchanged).unwrap();
            }
            panic!("seed hook");
        };
        let f = move |rt: &Runtime<C>| rt.get(x) * 10;
        let mut bag = Deferred::new();
        let caught = catch_unwind(AssertUnwindSafe(|| {
            get_or_insert_query_seeded(&table, &rt, 0, f, Default::default(), hook, &mut bag)
        }));
        bag.drain(&rt);
        assert!(caught.is_err());
        let q = created.get().expect("the miss created a node");
        assert!(!rt.is_alive(q));
        assert_eq!(rt.node_count(), nodes);
        assert_eq!(probe_unchecked(&table, &rt, &0), None);
        rt.assert_invariant_i();
        rt.assert_edges_complete();
        let again = table.get_or_insert_query(&rt, 0, f);
        assert_eq!(rt.get(again), 40);
    }
}

/// A key whose `Drop` panics once for the id `KEY_DROP_ARMED` holds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DropKey(u64);

static KEY_DROP_ARMED: AtomicU64 = AtomicU64::new(u64::MAX);

impl Drop for DropKey {
    fn drop(&mut self) {
        if KEY_DROP_ARMED
            .compare_exchange(self.0, u64::MAX, SeqCst, SeqCst)
            .is_ok()
        {
            panic!("drop of DropKey({}) panicked on purpose", self.0);
        }
    }
}

/// A seeded miss that rehashes leaves the old array's keys to the
/// caller's bag: a key's `Drop` panic surfaces at its drain, and the
/// node, already in the table, stays alive with its seed.
fn key_drop_panic_after_a_seeded_insert<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(4_u64);
    let table: NodeTable<DropKey, u64, C> = rt.create_table();
    let f = move |rt: &Runtime<C>| rt.get(x) * 10;
    for i in 0..14 {
        table.get_or_insert_query(&rt, DropKey(i), f);
    }
    assert_eq!(table.shape().0, 16);
    KEY_DROP_ARMED.store(3, SeqCst);
    let created = Cell::new(None);
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let mut bag = Deferred::new();
        get_or_insert_query_seeded(
            &table,
            &rt,
            DropKey(99),
            f,
            Default::default(),
            |rt, q| {
                created.set(Some(q));
                let base = seed_base(rt);
                seed_query(rt, &base, q, 41, &[dep(x)], Reuse::IfUnchanged)
                    .err()
                    .map(|e| e.1)
            },
            &mut bag,
        );
        assert!(created.get().is_some_and(|q| rt.is_alive(q)));
        bag.drain(&rt);
    }));
    let unwind = caught.expect_err("the key's drop panic surfaces");
    let message = unwind.downcast_ref::<String>().expect("a formatted panic");
    assert!(message.contains("DropKey(3)"), "{message}");
    assert_eq!(table.shape().0, 32);
    let q = created.get().expect("the miss created a node");
    assert!(rt.is_alive(q));
    assert_eq!(probe_unchecked(&table, &rt, &DropKey(99)), Some(q));
    assert_eq!(rt.state_of(q), NodeState::Dirty);
    assert_eq!(rt.get(q), 41);
    rt.assert_invariant_i();
    rt.assert_edges_complete();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_key_drop_panic_after_a_seeded_insert_keeps_the_node() {
    run_in_child(
        "runtime::seed_tests::local_key_drop_panic_after_a_seeded_insert_keeps_the_node",
        key_drop_panic_after_a_seeded_insert::<Local>,
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_key_drop_panic_after_a_seeded_insert_keeps_the_node() {
    run_in_child(
        "runtime::seed_tests::shared_key_drop_panic_after_a_seeded_insert_keeps_the_node",
        || {
            reclaim_probe::force(RetireSite::TableRetire);
            key_drop_panic_after_a_seeded_insert::<Shared>();
        },
    )
    .assert_clean();
}

/// A key that counts its drops; equal by its id alone.
#[derive(Clone)]
struct Noted(u8, Arc<AtomicU32>);

impl PartialEq for Noted {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Noted {}

impl std::hash::Hash for Noted {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl Drop for Noted {
    fn drop(&mut self) {
        self.1.fetch_add(1, SeqCst);
    }
}

/// A hit keeps neither its key nor its closure, and both wait in the
/// caller's bag until the drain.
fn a_seeded_hit_leaves_its_key_and_closure_in_the_bag<C: Cells>() {
    let rt = Runtime::<C>::new();
    let drops = Arc::new(AtomicU32::new(0));
    let key = || Noted(0, drops.clone());
    let table: NodeTable<Noted, u64, C> = rt.create_table();
    let q = table.get_or_insert_query(&rt, key(), |_: &Runtime<C>| 1);
    let held = key();
    let f = move |_: &Runtime<C>| {
        let held = &held;
        u64::from(held.0)
    };
    let mut bag = Deferred::new();
    let again = get_or_insert_query_seeded(
        &table,
        &rt,
        key(),
        f,
        Default::default(),
        |_, _| panic!("the hook ran on a hit"),
        &mut bag,
    );
    assert_eq!(again, q);
    assert_eq!(drops.load(SeqCst), 0);
    bag.drain(&rt);
    assert_eq!(drops.load(SeqCst), 2);
}

fn changed_at_erased_refuses_dead_and_foreign_handles<C: Cells>() {
    let rt = Runtime::<C>::new();
    let other = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    rt.set(x, 2);
    assert_eq!(changed_at_erased(&rt, x.erase()), Some(changed_at(&rt, x)));
    assert_eq!(changed_at_erased(&other, x.erase()), None);
    rt.delete_node(x);
    assert_eq!(changed_at_erased(&rt, x.erase()), None);
    let recycled = rt.create_input(3_u64);
    assert_eq!(recycled.slot(), x.slot());
    assert_eq!(changed_at_erased(&rt, x.erase()), None);
}

/// Outside a compute and in the retry nothing is doomed; in the doomed
/// run the pending handle reads the placeholder.
fn a_doomed_frame_sees_frame_doomed_and_reads_the_pending_handle<C: Cells>() {
    let rt = Runtime::<C>::new();
    rt.set_placeholder::<u64>(9);
    let gate = rt.doom_gate();
    assert!(!frame_doomed(&rt));
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let q = rt.create_query_with(
        move |rt| {
            let open = rt.get(gate).0;
            let doomed = frame_doomed(rt);
            let read = doomed.then(|| rt.get(pending_handle::<u64, _>(rt)));
            log.lock().unwrap().push((open, doomed, read));
            0_u64
        },
        super::replay_tests::replayable(),
    );
    rt.get(q);
    if !C::SINGLE_THREADED {
        // A Shared doomed frame unwinds at the gate's placeholder read, so
        // only the retry logs.
        assert_eq!(*seen.lock().unwrap(), [(true, false, None)]);
        return;
    }
    assert_eq!(
        *seen.lock().unwrap(),
        [(false, true, Some(9)), (true, false, None)]
    );
}

/// A dep seeded at W0 and trusted by R, claimed by a reader inside the
/// seed of R's `inner` hold, before its trust check (`before`) or after.
/// The reader loaded the closure before the hold and parks just before
/// its claim CAS, which takes no lock. Before, the check finds the dep
/// claimed or moved and R falls back to `RecomputeFirst`; after, the claim
/// reads from the check, so the reader's landing stamps above W0 and R
/// runs. Either way R reads the dep's new value.
fn trust_check_against_a_claim(before: bool) {
    // A forced revisit releases the reader's claim and parks it at the
    // claim again, so the schedule never forms.
    if super::forced_revisit() {
        return;
    }
    let rt = Arc::new(Rt::new());
    let y = rt.create_input(1_u64);
    let d = rt.create_query(move |rt: &Rt| rt.get(y) + 100);
    let (r, r_runs) = counted(&rt, d);
    let base = seed_base(&rt);
    assert_eq!(
        seed_query(&rt, &base, d, 7, &[dep(y)], Reuse::RecomputeFirst),
        Ok(Reuse::RecomputeFirst)
    );
    let (claim, trust) = (Gate::new(), Gate::new());
    let point = if before {
        ParkPoint::SeedBeforeTrust(id(d))
    } else {
        ParkPoint::SeedAfterTrust(id(d))
    };
    park_at(
        &rt,
        vec![
            (ParkPoint::BeforeClaim(id(d)), claim.clone()),
            (point, trust.clone()),
        ],
    );
    let reader = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || rt.get(d))
    };
    claim.wait_parked();
    let seeder = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || {
            seed_query(
                &rt,
                &base,
                r,
                70,
                &[trusted(d, base.w0)],
                Reuse::IfUnchanged,
            )
        })
    };
    trust.wait_parked();
    claim.release();
    let deadline = Instant::now() + Duration::from_secs(10);
    while rt.state_of(d) == NodeState::Dirty {
        assert!(Instant::now() < deadline, "the reader never claimed d");
        thread::yield_now();
    }
    trust.release();
    let seeded = seeder.join().unwrap();
    assert_eq!(reader.join().unwrap(), 101);
    assert_eq!(
        seeded,
        Ok(if before {
            Reuse::RecomputeFirst
        } else {
            Reuse::IfUnchanged
        })
    );
    assert_eq!(rt.get(r), 1010);
    assert_eq!(r_runs.load(SeqCst), 1);
    rt.assert_invariant_i();
}

#[test]
fn a_claim_before_the_trust_check_downgrades_the_seed() {
    trust_check_against_a_claim(true);
}

#[test]
fn a_claim_after_the_trust_check_stamps_above_w0() {
    trust_check_against_a_claim(false);
}

/// Q reads x and then a or b by `pick`, with a == b. A cutoff recompute
/// between `inspect`'s two holds moves Q from {x, a} to {x, b} without
/// moving its stamp, and `inspect` returns `None`.
#[test]
fn inspect_refuses_a_dep_change_between_its_holds() {
    let rt = Arc::new(Rt::new());
    let x = rt.create_input(1_u64);
    let pick = rt.create_input(false);
    let a = rt.create_input(5_u64);
    let b = rt.create_input(5_u64);
    let q = rt.create_query(move |rt: &Rt| {
        let v = rt.get(x);
        let side = if rt.get(pick) { b } else { a };
        v + rt.get(side)
    });
    assert_eq!(rt.get(q), 6);
    let stamp = rt.nodes.get(q.slot()).changed_at();
    let gate = Gate::new();
    park_at(
        &rt,
        vec![(ParkPoint::InspectBetweenHolds(id(q)), gate.clone())],
    );
    let inspector = {
        let rt = Arc::clone(&rt);
        thread::spawn(move || inspect(&rt, q).map(|s| (s.value, s.deps)))
    };
    gate.wait_parked();
    rt.set(pick, true);
    assert_eq!(rt.get(q), 6);
    assert_eq!(rt.nodes.get(q.slot()).changed_at(), stamp, "not a cutoff");
    gate.release();
    assert_eq!(inspector.join().unwrap(), None);
    let seen = inspect(&rt, q).expect("Clean");
    assert_eq!(
        seen.deps,
        vec![x.erase(), pick.erase(), b.erase()] as Vec<ErasedIncr>
    );
}
