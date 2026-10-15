//! Loom models for the `Shared` verification protocol (plan fix 8). Each
//! model runs the real runtime on the [`Loom`] strategy and, at the end
//! of every explored execution, checks the structural invariant (I) and
//! the value invariant (V) against an oracle.
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=3 LOOM_MAX_BRANCHES=100000 cargo test -p incr-core --release --lib loom_`

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;

use loom::sync::Arc;
use loom::thread;

use crate::loom_cells::Loom;
use crate::runtime::Runtime;
use crate::state::NodeState;

type Rt = Runtime<Loom>;

/// Runtime panics that a model expects on a racing thread (stale handle,
/// read of a deleted node, read of a cleared slot) would otherwise print
/// once per explored execution. Everything else, including the models'
/// own assertion failures, still reaches the default hook.
fn silence_expected_panics() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info
                .payload()
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| info.payload().downcast_ref::<&str>().copied())
                .unwrap_or("");
            let expected =
                msg.starts_with("incr-core:") || msg.starts_with("ValueSlot::read on a slot");
            if !expected {
                default(info);
            }
        }));
    });
}

fn run(f: impl Fn() + Sync + Send + 'static) {
    silence_expected_panics();
    loom::model(f);
}

/// x -> Q (early cutoff) -> D. One setter performs two sets, a second
/// setter one, and a reader verifies D. Targets B4, B5 and B6: a reader
/// that lands D `Clean` past a concurrent set leaves D `Clean` over a
/// non-Clean Q, or stale against x.
#[test]
fn loom_chain_cutoff() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(x) / 2);
        let d = rt.create_query(move |rt: &Rt| rt.get(q) + 100);
        assert_eq!(rt.get(d), 100);

        let setter_a = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.set(x, 1);
                rt.set(x, 2);
            })
        };
        let setter_b = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 3))
        };
        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(d);
            })
        };
        setter_a.join().unwrap();
        setter_b.join().unwrap();
        reader.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        let xv = rt.get(x);
        assert_eq!(
            rt.get(d),
            xv / 2 + 100,
            "(V) violated: D stale against x = {xv}"
        );
    });
}

/// x -> Q (early cutoff) -> {D1, D2}, one setter and one reader per
/// dependent. The shape that reaches fix 1's state-then-changed_at load
/// order: a verifier of D1 observes Q while the reader of D2 recomputes
/// it.
#[test]
fn loom_fan_out() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(x) / 2);
        let d1 = rt.create_query(move |rt: &Rt| rt.get(q) + 100);
        let d2 = rt.create_query(move |rt: &Rt| rt.get(q) * 10);
        assert_eq!(rt.get(d1), 100);
        assert_eq!(rt.get(d2), 0);

        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.set(x, 1);
                rt.set(x, 2);
            })
        };
        let r1 = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(d1);
            })
        };
        let r2 = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                rt.get(d2);
            })
        };
        setter.join().unwrap();
        r1.join().unwrap();
        r2.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        let qv = rt.get(x) / 2;
        assert_eq!(rt.get(d1), qv + 100, "(V) violated: D1 stale");
        assert_eq!(rt.get(d2), qv * 10, "(V) violated: D2 stale");
    });
}

/// Q's first compute reads v while another thread deletes v and recycles
/// its slot into a new input. Targets B1 and B2. The reader may unwind
/// (stale handle, `Dead` arm, cleared slot); when its `get` returns, the
/// value must be the oracle's. No edge may land on the new occupant, and
/// a successful delete means Q can't be `Clean` over v.
#[test]
fn loom_compute_races_delete() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let v = rt.create_input(7_u64);
        let q = rt.create_query(move |rt: &Rt| rt.get(v) + 1);

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || catch_unwind(AssertUnwindSafe(|| rt.get(q))).ok())
        };
        let deleter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                let deleted = catch_unwind(AssertUnwindSafe(|| rt.delete_node(v))).is_ok();
                let recycled = deleted.then(|| rt.create_input(99_u64));
                (deleted, recycled)
            })
        };
        let read = reader.join().unwrap();
        let (deleted, recycled) = deleter.join().unwrap();

        if let Some(value) = read {
            assert_eq!(
                value, 8,
                "(V) violated: Q returned a value not computed from v"
            );
        }
        rt.assert_invariant_i();
        rt.assert_walks_complete();
        if let Some(new) = recycled {
            assert!(
                !rt.has_reverse_edge(new, q),
                "phantom edge: Q landed on the recycled slot's dependents"
            );
        }
        if deleted {
            assert_ne!(
                rt.state_of(q),
                NodeState::Clean,
                "Q is Clean over a deleted dep"
            );
        }
    });
}

/// A dependent P over a query Q whose first compute races a delete of v,
/// while a setter moves x so that Q stops reading v. Exercises the stale
/// recovery (fix 5) and the walk through a recovered node (fix 6): once
/// x = 1 every path must recover, so the final read of P is exact.
#[test]
fn loom_recovered_node_under_set() {
    run(|| {
        let rt = Arc::new(Rt::new());
        let x = rt.create_input(0_u64);
        let v = rt.create_input(7_u64);
        let q = rt.create_query(move |rt: &Rt| {
            let xv = rt.get(x);
            if xv == 0 {
                xv + rt.get(v)
            } else {
                xv
            }
        });
        let p = rt.create_query(move |rt: &Rt| rt.get(q) + 100);

        let reader = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                let _ = catch_unwind(AssertUnwindSafe(|| rt.get(p)));
            })
        };
        let deleter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || {
                let deleted = catch_unwind(AssertUnwindSafe(|| rt.delete_node(v))).is_ok();
                deleted.then(|| rt.create_input(99_u64))
            })
        };
        let setter = {
            let rt = Arc::clone(&rt);
            thread::spawn(move || rt.set(x, 1))
        };
        reader.join().unwrap();
        let recycled = deleter.join().unwrap();
        setter.join().unwrap();

        rt.assert_invariant_i();
        rt.assert_walks_complete();
        if let Some(new) = recycled {
            assert!(
                !rt.has_reverse_edge(new, q),
                "phantom edge: Q landed on the recycled slot's dependents"
            );
        }
        assert_eq!(
            rt.get(p),
            101,
            "(V) violated: P did not recover after x = 1"
        );
    });
}
