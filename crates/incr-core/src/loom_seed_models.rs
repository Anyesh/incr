//! Loom models for the restore and checkpoint hooks (T10): `seed_query`
//! racing a set, a recompute of a trusted dep and a reader, and `inspect`
//! racing a cutoff recompute that changes deps. A loom `RwLock` stands in
//! for incr-persist's `DepIndex`: a mapped write sets the input and its
//! fingerprint under the write hold, and a restore reads the fingerprint
//! under the read hold after taking its base.
//!
//! Run one with:
//! `RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=2 LOOM_MAX_BRANCHES=100000 cargo test -p incr-core --release --lib -- --exact loom_seed_models::seed_vs_set`

use loom::sync::{Arc, RwLock};
use loom::thread;

use crate::handle::Incr;
use crate::loom_cells::Loom;
use crate::loom_models::run;
use crate::persist_hooks::{inspect, seed_base, seed_query, Reuse, SeedDep};
use crate::runtime::Runtime;

type Rt = Runtime<Loom>;

fn dep<T>(h: Incr<T>) -> SeedDep {
    SeedDep {
        dep: h.erase(),
        trusted_stamp: None,
    }
}

/// A restore of q = x * 10, recorded at x = 0, races a mapped write of
/// x. The restore trusts x only if the fingerprint it reads after its
/// base still says 0; either way q ends equal to x * 10.
#[test]
fn seed_vs_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(x) * 10);
        let fp = Arc::new(RwLock::new(0_u64));
        let writer = {
            let (rt, fp) = (Arc::clone(&rt), Arc::clone(&fp));
            thread::spawn(move || {
                let mut fp = fp.write().unwrap();
                rt.set(x, 1);
                *fp = 1;
            })
        };
        let restorer = {
            let (rt, fp) = (Arc::clone(&rt), Arc::clone(&fp));
            thread::spawn(move || {
                let base = seed_base(&rt);
                let reuse = if *fp.read().unwrap() == 0 {
                    Reuse::IfUnchanged
                } else {
                    Reuse::RecomputeFirst
                };
                seed_query(&rt, &base, q, 0, &[dep(x)], reuse).unwrap();
            })
        };
        writer.join().unwrap();
        restorer.join().unwrap();
        rt.assert_invariant_i();
        rt.assert_edges_complete();
        assert_eq!(rt.get(q), rt.get(x) * 10, "a restore reused a stale value");
    });
}

/// d = x + 100 is seeded at W0 with its value; r = d * 10 is seeded
/// `IfUnchanged` trusting d at W0 while another thread sets x and reads
/// d, whose recompute changes it. r ends equal to (x + 100) * 10.
#[test]
fn seed_vs_dep_recompute() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(1_u64);
        let d = rt.create_query(move |rt: &Rt| rt.get(x) + 100);
        let r = rt.create_query(move |rt: &Rt| rt.get(d) * 10);
        let base = seed_base(&rt);
        seed_query(&rt, &base, d, 101, &[dep(x)], Reuse::RecomputeFirst).unwrap();
        let other = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.set(x, 2);
                rt.get(d)
            })
        };
        let trust = SeedDep {
            dep: d.erase(),
            trusted_stamp: Some(base.w0),
        };
        seed_query(&rt, &base, r, 1010, &[trust], Reuse::IfUnchanged).unwrap();
        assert_eq!(other.join().unwrap(), 102);
        rt.assert_invariant_i();
        rt.assert_edges_complete();
        assert_eq!(rt.get(r), 1020, "r verified over a recomputed d");
    });
}

/// A reader of q races its restore: the reader either computes q cold,
/// and the seed finds it not fresh, or reads the seeded node. Both see
/// x * 10.
#[test]
fn seed_vs_reader() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(3_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(x) * 10);
        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.get(q))
        };
        let base = seed_base(&rt);
        let _ = seed_query(&rt, &base, q, 30, &[dep(x)], Reuse::IfUnchanged);
        assert_eq!(reader.join().unwrap(), 30);
        rt.assert_invariant_i();
        rt.assert_edges_complete();
        assert_eq!(rt.get(q), 30);
    });
}

/// Q = x + (a or b by pick), with b != a and, for the value-preserving
/// dep change, b == a. A recompute moves Q from {pick, x, a} to
/// {pick, x, b} while `inspect` reads it: what `inspect` returns is one
/// run's value with that run's deps.
#[test]
fn inspect_vs_recompute() {
    for b_value in [7_u64, 5] {
        run(move || {
            let rt = Arc::new(Rt::new());
            let x = rt.create_input(1_u64);
            let pick = rt.create_input(false);
            let a = rt.create_input(5_u64);
            let b = rt.create_input(b_value);
            let q = rt.create_query(move |rt: &Rt| {
                let side = if rt.get(pick) { b } else { a };
                rt.get(x) + rt.get(side)
            });
            assert_eq!(rt.get(q), 6);
            let other = {
                let rt = Arc::clone(&rt);
                thread::spawn(move || {
                    rt.set(pick, true);
                    rt.get(q)
                })
            };
            let seen = inspect(&rt, q);
            assert_eq!(other.join().unwrap(), 1 + b_value);
            if let Some(seen) = seen {
                let before = (6, vec![pick.erase(), x.erase(), a.erase()]);
                let after = (1 + b_value, vec![pick.erase(), x.erase(), b.erase()]);
                let pair = (seen.value, seen.deps);
                assert!(
                    pair == before || pair == after,
                    "inspect mixed two runs: {pair:?}"
                );
            }
            rt.assert_invariant_i();
        });
    }
}
