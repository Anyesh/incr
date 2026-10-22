//! The value-code contract: a user `Clone`, `PartialEq`, `Hash` or `Eq`
//! the engine calls may read Clean nodes and must not otherwise call
//! back into the runtime. Each case runs under `Local` and `Shared`,
//! through a hook that the chosen trait method of `Hooked` fires on its
//! n-th call.

use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::drop_panic_tests::prime_reclaim_timer;
use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::child_test::run_in_child;
use crate::persist_hooks::{seed_base, seed_query, Reuse, SeedDep};
use crate::reclaim_probe::{self, RetireSite};
use crate::set_stack;
use crate::state::NodeState;
use crate::value_code::CONTRACT_MESSAGE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Clone,
    Eq,
    Hash,
    Drop,
}

struct Hook {
    on: Op,
    nth: u32,
    f: Box<dyn FnMut()>,
}

thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    static CALLS: Cell<u32> = const { Cell::new(0) };
}

/// Arms `f` to run on the `nth` call of `on`. A hook that panics stays
/// consumed, so the calls after the panic run no user code.
fn arm_nth(on: Op, nth: u32, f: impl FnMut() + 'static) {
    CALLS.with(|c| c.set(0));
    HOOK.with(|h| {
        *h.borrow_mut() = Some(Hook {
            on,
            nth,
            f: Box::new(f),
        })
    });
}

fn arm(on: Op, f: impl FnMut() + 'static) {
    arm_nth(on, 1, f);
}

fn disarm() {
    HOOK.with(|h| *h.borrow_mut() = None);
}

fn calls() -> u32 {
    CALLS.with(Cell::get)
}

fn fire(op: Op) {
    // `try_with`, because a hook's captures hold values whose `Drop`
    // fires again while the thread-locals are being destroyed.
    let taken = HOOK.try_with(|h| {
        let mut h = h.borrow_mut();
        match h.as_mut() {
            Some(hook) if hook.on == op => {
                hook.nth -= 1;
                (hook.nth == 0).then(|| h.take().unwrap())
            }
            _ => None,
        }
    });
    if let Ok(Some(mut hook)) = taken {
        CALLS.with(|c| c.set(c.get() + 1));
        (hook.f)();
        hook.nth = 1;
        HOOK.with(|h| *h.borrow_mut() = Some(hook));
    }
}

#[derive(Debug)]
pub(crate) struct Hooked(pub(crate) u32);

impl Clone for Hooked {
    fn clone(&self) -> Self {
        fire(Op::Clone);
        Hooked(self.0)
    }
}

impl PartialEq for Hooked {
    fn eq(&self, other: &Self) -> bool {
        fire(Op::Eq);
        self.0 == other.0
    }
}

impl Eq for Hooked {}

impl Hash for Hooked {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fire(Op::Hash);
        self.0.hash(state);
    }
}

impl Drop for Hooked {
    fn drop(&mut self) {
        fire(Op::Drop);
    }
}

fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
    match err.downcast::<String>() {
        Ok(s) => *s,
        Err(err) => err.downcast::<&str>().map(|s| s.to_string()).unwrap(),
    }
}

/// Runs `op`, which must panic with the contract message from the first
/// hooked call, and leaves the hook disarmed.
fn assert_contract_panic(what: &str, op: impl FnOnce()) {
    let err = catch_unwind(AssertUnwindSafe(op))
        .expect_err(&format!("{what}: the runtime call was allowed"));
    let message = panic_message(err);
    assert!(
        message.contains(CONTRACT_MESSAGE),
        "{what}: expected the contract panic, got: {message}"
    );
    assert_eq!(
        calls(),
        1,
        "{what}: the panic did not come from the first hooked call"
    );
    disarm();
}

fn ids<C: Cells>(rt: &Runtime<C>, c: &crate::IncrCollection<Hooked, C>) -> Vec<u32> {
    let mut out: Vec<u32> = c.values(rt).iter().map(|h| h.0).collect();
    out.sort_unstable();
    out
}

/// `set` compares once, under its claim.
const SET_COMPARES: std::ops::RangeInclusive<u32> = 1..=1;

fn partial_eq_that_deletes_the_input<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    for nth in SET_COMPARES {
        let hooked = Rc::clone(&rt);
        arm_nth(Op::Eq, nth, move || hooked.delete_node(a));
        assert_contract_panic("delete from PartialEq", || rt.set(a, Hooked(2)));
        assert!(rt.is_alive(a));
        assert_eq!(rt.state_of(a), NodeState::Clean);
        assert_eq!(rt.get(a).0, 1);
    }
    rt.set(a, Hooked(3));
    assert_eq!(rt.get(a).0, 3);
}

fn partial_eq_that_sets_another_input<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let b = rt.create_input(10u32);
    for nth in SET_COMPARES {
        let hooked = Rc::clone(&rt);
        arm_nth(Op::Eq, nth, move || hooked.set(b, 11));
        assert_contract_panic("set from PartialEq", || rt.set(a, Hooked(2)));
        assert_eq!(rt.get(b), 10);
        assert_eq!(rt.state_of(a), NodeState::Clean);
        assert_eq!(rt.get(a).0, 1);
    }
    rt.set(a, Hooked(3));
    assert_eq!(rt.get(a).0, 3);
}

fn partial_eq_that_inserts_into_a_collection<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let c = rt.create_collection::<u32>();
    for nth in SET_COMPARES {
        let hooked = Rc::clone(&rt);
        let target = c.clone();
        arm_nth(Op::Eq, nth, move || target.insert(&hooked, 7));
        assert_contract_panic("insert from PartialEq", || rt.set(a, Hooked(2)));
        assert_eq!(c.snapshot_len(), 0);
        assert_eq!(rt.state_of(a), NodeState::Clean);
        assert_eq!(rt.get(a).0, 1);
    }
    c.insert(&rt, 7);
    assert_eq!(c.values(&rt), vec![7]);
}

/// A `PartialEq` that panics on either of `set`'s compares leaves the
/// input Clean on its old value, and the next set goes through.
fn partial_eq_that_panics_during_set<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    for nth in SET_COMPARES {
        arm_nth(Op::Eq, nth, || panic!("eq bomb"));
        let err =
            catch_unwind(AssertUnwindSafe(|| rt.set(a, Hooked(2)))).expect_err("set returned");
        assert!(panic_message(err).contains("eq bomb"));
        disarm();
        assert_eq!(rt.state_of(a), NodeState::Clean);
        assert_eq!(rt.get(a).0, 1);
    }
    rt.set(a, Hooked(3));
    assert_eq!(rt.get(a).0, 3);
}

/// A `Clone` that panics in a read and a `PartialEq` that panics in a
/// set and in a recompute each leave the value-code count balanced, so
/// the same runtime still sets, reads and computes afterwards.
fn panicking_value_code_leaves_the_runtime_usable<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let q = rt.create_query(move |rt| Hooked(rt.get(a).0 + 10));
    assert_eq!(rt.get(q).0, 11);

    arm(Op::Clone, || panic!("clone bomb"));
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(a))).expect_err("get returned");
    assert!(panic_message(err).contains("clone bomb"));
    disarm();
    assert_eq!(crate::value_code::thread_word(), 0);

    arm(Op::Eq, || panic!("eq bomb"));
    let err = catch_unwind(AssertUnwindSafe(|| rt.set(a, Hooked(2)))).expect_err("set returned");
    assert!(panic_message(err).contains("eq bomb"));
    disarm();
    assert_eq!(crate::value_code::thread_word(), 0);
    assert_eq!(rt.get(a).0, 1);

    rt.set(a, Hooked(2));
    arm(Op::Eq, || panic!("eq bomb"));
    let err = catch_unwind(AssertUnwindSafe(|| rt.get(q))).expect_err("recompute returned");
    assert!(panic_message(err).contains("eq bomb"));
    disarm();
    assert_eq!(crate::value_code::thread_word(), 0);

    rt.set(a, Hooked(3));
    assert_eq!(rt.get(a).0, 3);
    assert_eq!(rt.get(q).0, 13);
    let b = rt.create_input(Hooked(4));
    assert_eq!(rt.get(b).0, 4);
}

/// A `Drop` that deletes its input while unwinding out of that input's
/// set is refused before the delete claims anything, so a `Drop` that
/// catches the refusal leaves the input alive, settable and deletable.
fn drop_that_deletes_the_input_under_its_set<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let refused = Rc::new(Cell::new(None));
    let hooked = Rc::clone(&rt);
    let record = Rc::clone(&refused);
    arm(Op::Eq, move || {
        let rt = Rc::clone(&hooked);
        let record = Rc::clone(&record);
        arm(Op::Drop, move || {
            let err = catch_unwind(AssertUnwindSafe(|| rt.delete_node(a)))
                .expect_err("delete under its own set returned");
            record.set(Some(panic_message(err)));
        });
        panic!("eq bomb");
    });
    let err = catch_unwind(AssertUnwindSafe(|| rt.set(a, Hooked(2)))).expect_err("set returned");
    assert_eq!(panic_message(err), "eq bomb");
    disarm();
    let refusal = refused
        .take()
        .expect("the incoming value's Drop did not run");
    assert!(refusal.contains("set of it was in progress"), "{refusal}");
    assert!(rt.is_alive(a));
    assert_eq!(rt.state_of(a), NodeState::Clean);
    assert_eq!(rt.get(a).0, 1);
    rt.set(a, Hooked(3));
    assert_eq!(rt.get(a).0, 3);
    rt.delete_node(a);
    assert!(!rt.is_alive(a));
}

/// A `Drop` of the value a `set` displaced runs after the claim's
/// release, the walk and the set-stack pop: the hook records that the
/// input is Clean on the new value and no longer held by this thread's
/// set stack. Its panic leaves `set` with the input Clean on the new
/// value, and the next set goes through. Under `Shared` the drop runs
/// at the retire's reclaim, which the child forces at that site.
fn drop_that_panics_during_set<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let d = rt.create_query(move |rt: &Runtime<C>| rt.get(a).0 + 10);
    assert_eq!(rt.get(d), 11);
    let seen = Rc::new(Cell::new(None));
    let hooked = Rc::clone(&rt);
    let record = Rc::clone(&seen);
    arm(Op::Drop, move || {
        let held = set_stack::holds(hooked.runtime_id, a.slot());
        record.set(Some((hooked.state_of(a), held, hooked.get(a).0)));
        panic!("drop bomb");
    });
    let err = catch_unwind(AssertUnwindSafe(|| rt.set(a, Hooked(2)))).expect_err("set returned");
    assert_eq!(panic_message(err), "drop bomb");
    assert_eq!(calls(), 1, "the displaced value's Drop did not run");
    assert_eq!(
        seen.get(),
        Some((NodeState::Clean, false, 2)),
        "the displaced value dropped before the set landed and popped"
    );
    disarm();
    assert_eq!(rt.state_of(a), NodeState::Clean);
    assert_eq!(rt.get(a).0, 2);
    assert_eq!(rt.get(d), 12);
    rt.set(a, Hooked(3));
    assert_eq!(rt.get(d), 13);
}

fn clone_that_reads_a_clean_node<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let b = rt.create_input(5u32);
    let q = rt.create_query(move |rt| rt.get(b) * 2);
    assert_eq!(rt.get(q), 10);
    let hooked = Rc::clone(&rt);
    arm(Op::Clone, move || {
        assert_eq!(hooked.get(b), 5);
        assert_eq!(hooked.get(q), 10);
    });
    assert_read_allowed("Clean get from Clone", || {
        assert_eq!(rt.get(a).0, 1);
    });
    rt.set(a, Hooked(2));
    assert_eq!(rt.get(a).0, 2);
}

/// A Clean read under `Local` skips the generation recheck after its
/// clone because the clone cannot delete, recycle or overwrite the node
/// it is cloning; each attempt panics and leaves the node as it was.
fn clone_that_deletes_or_sets_the_node_it_reads<C: Cells>(rt: Rc<Runtime<C>>) {
    type Change<C> = (&'static str, fn(&Runtime<C>, crate::Incr<Hooked>));
    let changes: [Change<C>; 4] = [
        ("delete_node", |rt, a| rt.delete_node(a)),
        ("try_delete_node", |rt, a| {
            let _ = rt.try_delete_node(a);
        }),
        ("delete_with_dependents", |rt, a| {
            rt.delete_with_dependents(a);
        }),
        ("set", |rt, a| rt.set(a, Hooked(2))),
    ];
    let a = rt.create_input(Hooked(1));
    for (name, change) in changes {
        for try_get in [false, true] {
            let hooked = Rc::clone(&rt);
            arm(Op::Clone, move || change(&hooked, a));
            assert_contract_panic(name, || {
                if try_get {
                    let _ = rt.try_get(a);
                } else {
                    rt.get(a);
                }
            });
            assert!(rt.is_alive(a), "{name}");
            assert_eq!(rt.state_of(a), NodeState::Clean, "{name}");
            assert_eq!(rt.get(a).0, 1, "{name}");
        }
    }
}

fn clone_that_creates_a_node<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let hooked = Rc::clone(&rt);
    arm(Op::Clone, move || {
        hooked.create_input(0u32);
    });
    let before = rt.node_count();
    assert_contract_panic("create from Clone", || {
        rt.get(a);
    });
    assert_eq!(rt.node_count(), before);
    assert_eq!(rt.get(a).0, 1);
}

/// A restore's `seed_query` from a Clean read's `Clone` is refused before
/// it claims the target or installs an edge, so the target stays New and
/// seeds normally afterwards.
fn clone_that_seeds_a_query<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let x = rt.create_input(3u64);
    let q = rt.create_query(move |rt| rt.get(x) * 10);
    let base = seed_base(&rt);
    let dep = SeedDep {
        dep: x.erase(),
        trusted_stamp: None,
    };
    let hooked = Rc::clone(&rt);
    arm(Op::Clone, move || {
        let _ = seed_query(&hooked, &base, q, 30, &[dep], Reuse::IfUnchanged);
    });
    assert_contract_panic("seed_query from Clone", || {
        rt.get(a);
    });
    assert!(rt.is_alive(a));
    assert_eq!(rt.state_of(a), NodeState::Clean);
    assert_eq!(rt.get(a).0, 1);
    assert_eq!(rt.state_of(q), NodeState::New);
    assert_eq!(rt.nodes.get(q.slot()).dep_count(), 0);
    assert!(rt.inner.read().dependents[x.slot() as usize].is_empty());
    assert_eq!(
        seed_query(&rt, &base, q, 30, &[dep], Reuse::IfUnchanged),
        Ok(Reuse::IfUnchanged)
    );
    assert_eq!(rt.get(q), 30);
    rt.set(x, 4);
    assert_eq!(rt.get(q), 40);
}

/// A read from the `PartialEq` of a recompute runs inside the query's
/// frame, so it records a dependency: the query recomputes when the node
/// it read changes.
fn partial_eq_of_a_recompute_that_reads_a_node<C: Cells>(rt: Rc<Runtime<C>>) {
    let x = rt.create_input(1u32);
    let other = rt.create_input(5u32);
    let runs = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&runs);
    let q = rt.create_query(move |rt| {
        counted.fetch_add(1, Ordering::Relaxed);
        Hooked(rt.get(x))
    });
    assert_eq!(rt.get(q).0, 1);
    rt.set(x, 2);
    let hooked = Rc::clone(&rt);
    arm(Op::Eq, move || {
        assert_eq!(hooked.get(other), 5);
    });
    assert_read_allowed("get from write_if_changed", || {
        assert_eq!(rt.get(q).0, 2);
    });
    assert_eq!(runs.load(Ordering::Relaxed), 2);
    rt.set(other, 6);
    assert_eq!(rt.get(q).0, 2);
    assert_eq!(
        runs.load(Ordering::Relaxed),
        3,
        "the read in PartialEq recorded no dependency"
    );
}

/// Runs `op`, whose hooked calls read Clean nodes, which must succeed,
/// and leaves the hook disarmed.
fn assert_read_allowed(what: &str, op: impl FnOnce()) {
    if let Err(err) = catch_unwind(AssertUnwindSafe(op)) {
        panic!("{what}: the read was refused: {}", panic_message(err));
    }
    assert!(calls() >= 1, "{what}: the hooked call did not run");
    disarm();
}

/// The nodes and collections the checked-path table calls into, built
/// before the hook is armed.
struct Fixture<C: Cells> {
    clean: crate::Incr<u32>,
    source: crate::Incr<u32>,
    dirty: crate::Incr<u32>,
    fresh: crate::Incr<u32>,
    victim: crate::Incr<u32>,
    observer: crate::ObserverId,
    c: crate::IncrCollection<u32, C>,
    grouped: crate::GroupedCollection<u32, u32, C>,
    sorted: crate::SortedCollection<u32, u32, C>,
}

impl<C: Cells> Fixture<C> {
    fn new(rt: &Runtime<C>) -> Self {
        let clean = rt.create_input(5u32);
        let source = rt.create_input(1u32);
        let dirty = rt.create_query(move |rt| rt.get(source) + 1);
        let fresh = rt.create_query(move |rt| rt.get(source) + 2);
        let victim = rt.create_input(0u32);
        let observer = rt.observe(clean, |_| {});
        let c = rt.create_collection::<u32>();
        let grouped = c.group_by(rt, |v| v % 2);
        let sorted = c.sort_by_key(rt, |v| *v);
        c.insert(rt, 1);
        rt.get(clean);
        rt.get(dirty);
        rt.get(grouped.version_node());
        rt.get(sorted.version_node());
        Self {
            clean,
            source,
            dirty,
            fresh,
            victim,
            observer,
            c,
            grouped,
            sorted,
        }
    }
}

type Path<C> = (&'static str, fn(&Runtime<C>, &Fixture<C>));

fn engine_paths<C: Cells>() -> Vec<Path<C>> {
    vec![
        ("get of a Dirty node", |rt, f| {
            rt.get(f.dirty);
        }),
        ("get of a New node", |rt, f| {
            rt.get(f.fresh);
        }),
        ("set", |rt, f| rt.set(f.victim, 1)),
        ("delete_node", |rt, f| rt.delete_node(f.victim)),
        ("try_delete_node", |rt, f| {
            let _ = rt.try_delete_node(f.victim);
        }),
        ("delete_with_dependents", |rt, f| {
            rt.delete_with_dependents(f.victim);
        }),
        ("create_input", |rt, _| {
            rt.create_input(0u32);
        }),
        ("create_query", |rt, _| {
            rt.create_query(|_| 0u32);
        }),
        ("observe", |rt, f| {
            rt.observe(f.clean, |_| {});
        }),
        ("unobserve", |rt, f| rt.unobserve(f.observer)),
        ("stabilize", |rt, _| rt.stabilize()),
        ("create_collection", |rt, _| {
            rt.create_collection::<u32>();
        }),
    ]
}

fn collection_paths<C: Cells>() -> Vec<Path<C>> {
    vec![
        ("insert", |rt, f| f.c.insert(rt, 2)),
        ("delete", |rt, f| {
            f.c.delete(rt, &1);
        }),
        ("replace", |rt, f| {
            f.c.replace(rt, Some(&1), 2);
        }),
        ("values", |rt, f| {
            f.c.values(rt);
        }),
        ("snapshot_len", |_, f| {
            f.c.snapshot_len();
        }),
        ("filter", |rt, f| {
            f.c.filter(rt, |_| true);
        }),
        ("map", |rt, f| {
            f.c.map(rt, |v| *v);
        }),
        ("count", |rt, f| {
            f.c.count(rt);
        }),
        ("reduce", |rt, f| {
            f.c.reduce(rt, |v| v.len());
        }),
        ("aggregate", |rt, f| {
            f.c.aggregate(rt, 0u32, |v| *v, |a, b| a + b);
        }),
        ("join", |rt, f| {
            f.c.join(rt, &f.c, |v| *v, |v| *v);
        }),
        ("group_by", |rt, f| {
            f.c.group_by(rt, |v| *v);
        }),
        ("keys", |_, f| {
            f.grouped.keys();
        }),
        ("get_group", |_, f| {
            f.grouped.get_group(&1);
        }),
        ("group_count", |_, f| {
            f.grouped.group_count();
        }),
        ("sort_by_key", |rt, f| {
            f.c.sort_by_key(rt, |v| *v);
        }),
        ("sorted snapshot", |_, f| {
            f.sorted.snapshot();
        }),
        ("sorted snapshot_len", |_, f| {
            f.sorted.snapshot_len();
        }),
        ("pairwise", |rt, f| {
            f.sorted.pairwise(rt);
        }),
        ("window", |rt, f| {
            f.sorted.window(rt, 2);
        }),
    ]
}

/// Each path, called from the `PartialEq` of a `set`, panics with the
/// contract message before changing anything, and the engine serves the
/// next call normally.
fn every_path_is_refused<C: Cells>(rt: Rc<Runtime<C>>, paths: Vec<Path<C>>) {
    let fixture = Rc::new(Fixture::new(&rt));
    for (i, (name, path)) in paths.into_iter().enumerate() {
        let a = rt.create_input(Hooked(1));
        rt.set(fixture.source, i as u32 + 2);
        let nodes = rt.node_count();
        let hooked = Rc::clone(&rt);
        let f = Rc::clone(&fixture);
        arm(Op::Eq, move || path(&hooked, &f));
        assert_contract_panic(name, || rt.set(a, Hooked(2)));
        assert_eq!(rt.get(a).0, 1, "{name}");
        assert_eq!(rt.node_count(), nodes, "{name}");
        assert_eq!(rt.get(fixture.victim), 0, "{name}");
        assert_eq!(fixture.c.values(&rt), vec![1], "{name}");
        rt.set(a, Hooked(3));
        assert_eq!(rt.get(a).0, 3, "{name}");
    }
}

fn hash_that_reads_a_node_on_insert<C: Cells>(rt: Rc<Runtime<C>>) {
    let b = rt.create_input(5u32);
    let c = rt.create_collection::<Hooked>();
    let hooked = Rc::clone(&rt);
    arm(Op::Hash, move || {
        assert_eq!(hooked.get(b), 5);
    });
    assert_read_allowed("get from Hash on insert", || c.insert(&rt, Hooked(1)));
    assert_eq!(ids(&rt, &c), vec![1]);
}

fn eq_that_reads_a_node_on_insert<C: Cells>(rt: Rc<Runtime<C>>) {
    let b = rt.create_input(5u32);
    let c = rt.create_collection::<Hooked>();
    c.insert(&rt, Hooked(1));
    let hooked = Rc::clone(&rt);
    arm(Op::Eq, move || {
        assert_eq!(hooked.get(b), 5);
    });
    assert_read_allowed("get from Eq on insert", || c.insert(&rt, Hooked(1)));
    assert_eq!(ids(&rt, &c), vec![1, 1]);
}

fn clone_that_reads_a_node_on_values<C: Cells>(rt: Rc<Runtime<C>>) {
    let b = rt.create_input(5u32);
    let c = rt.create_collection::<Hooked>();
    c.insert(&rt, Hooked(1));
    let hooked = Rc::clone(&rt);
    arm(Op::Clone, move || {
        assert_eq!(hooked.get(b), 5);
    });
    assert_read_allowed("get from Clone on values", || {
        c.values(&rt);
    });
    assert_eq!(ids(&rt, &c), vec![1]);
}

fn clone_that_reads_a_node_in_a_filter_batch<C: Cells>(rt: Rc<Runtime<C>>) {
    let b = rt.create_input(5u32);
    let c = rt.create_collection::<Hooked>();
    let evens = c.filter(&rt, |h| h.0 % 2 == 0);
    c.insert(&rt, Hooked(2));
    let hooked = Rc::clone(&rt);
    arm(Op::Clone, move || {
        assert_eq!(hooked.get(b), 5);
    });
    assert_read_allowed("get from Clone in a filter batch", || {
        rt.get(evens.version_node());
    });
    c.insert(&rt, Hooked(4));
    assert_eq!(ids(&rt, &evens), vec![2, 4]);
}

fn hash_that_reads_a_node_in_a_group_by_batch<C: Cells>(rt: Rc<Runtime<C>>) {
    let b = rt.create_input(5u32);
    let c = rt.create_collection::<Hooked>();
    let grouped = c.group_by(&rt, |h| h.0 % 2);
    c.insert(&rt, Hooked(2));
    let hooked = Rc::clone(&rt);
    arm(Op::Hash, move || {
        assert_eq!(hooked.get(b), 5);
    });
    assert_read_allowed("get from Hash in the group_by key cache", || {
        rt.get(grouped.version_node());
    });
    c.insert(&rt, Hooked(3));
    rt.get(grouped.version_node());
    assert_eq!(ids(&rt, &grouped.get_group(&0).unwrap()), vec![2]);
    assert_eq!(ids(&rt, &grouped.get_group(&1).unwrap()), vec![3]);
}

fn hash_that_reads_a_node_in_a_join_batch_and_its_rebuild<C: Cells>(rt: Rc<Runtime<C>>) {
    let b = rt.create_input(5u32);
    let left = rt.create_collection::<Hooked>();
    let right = rt.create_collection::<Hooked>();
    let joined = left.join(&rt, &right, |h| h.0, |h| h.0);
    left.insert(&rt, Hooked(1));
    right.insert(&rt, Hooked(1));
    let hooked = Rc::clone(&rt);
    arm(Op::Hash, move || {
        assert_eq!(hooked.get(b), 5);
    });
    assert_read_allowed("get from Hash in the join key cache", || {
        rt.get(joined.version_node());
    });
    right.insert(&rt, Hooked(2));
    left.insert(&rt, Hooked(2));
    let mut pairs: Vec<(u32, u32)> = joined.values(&rt).iter().map(|(l, r)| (l.0, r.0)).collect();
    pairs.sort_unstable();
    assert_eq!(pairs, vec![(1, 1), (2, 2)]);
}

fn clone_that_reads_a_node_in_a_sort_batch<C: Cells>(rt: Rc<Runtime<C>>) {
    let b = rt.create_input(5u32);
    let c = rt.create_collection::<Hooked>();
    let sorted = c.sort_by_key(&rt, |h| h.0);
    let pairs = sorted.pairwise(&rt);
    c.insert(&rt, Hooked(2));
    // The first clone stages the pending delta, the second files the
    // element in the sorted view.
    let hooked = Rc::clone(&rt);
    arm_nth(Op::Clone, 2, move || {
        assert_eq!(hooked.get(b), 5);
    });
    assert_read_allowed("get from Clone in the sorted view", || {
        rt.get(sorted.version_node());
    });
    c.insert(&rt, Hooked(1));
    rt.get(sorted.version_node());
    let hooked = Rc::clone(&rt);
    arm(Op::Clone, move || {
        assert_eq!(hooked.get(b), 5);
    });
    assert_read_allowed("get from Clone in a pairwise batch", || {
        rt.get(pairs.version_node());
    });
    c.insert(&rt, Hooked(3));
    let mut out: Vec<(u32, u32)> = pairs.values(&rt).iter().map(|(l, r)| (l.0, r.0)).collect();
    out.sort_unstable();
    assert_eq!(out, vec![(1, 2), (2, 3)]);
}

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        $(
            mod $name {
                use super::*;

                #[test]
                fn local() {
                    super::$name(Rc::new(Runtime::<Local>::new()));
                }

                #[test]
                fn shared() {
                    super::$name(Rc::new(Runtime::<Shared>::new()));
                }
            }
        )*
    };
}

both!(
    partial_eq_that_deletes_the_input,
    partial_eq_that_sets_another_input,
    partial_eq_that_inserts_into_a_collection,
    partial_eq_that_panics_during_set,
    panicking_value_code_leaves_the_runtime_usable,
    drop_that_deletes_the_input_under_its_set,
    clone_that_reads_a_clean_node,
    clone_that_deletes_or_sets_the_node_it_reads,
    clone_that_creates_a_node,
    clone_that_seeds_a_query,
    partial_eq_of_a_recompute_that_reads_a_node,
    hash_that_reads_a_node_on_insert,
    eq_that_reads_a_node_on_insert,
    clone_that_reads_a_node_on_values,
    clone_that_reads_a_node_in_a_filter_batch,
    hash_that_reads_a_node_in_a_group_by_batch,
    hash_that_reads_a_node_in_a_join_batch_and_its_rebuild,
    clone_that_reads_a_node_in_a_sort_batch,
);

mod drop_that_panics_during_set {
    use super::*;

    #[test]
    fn local() {
        super::drop_that_panics_during_set(Rc::new(Runtime::<Local>::new()));
    }

    /// In a child, because the forced reclaim would otherwise drop the
    /// armed value on whichever test thread retires next.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared() {
        run_in_child(
            "runtime::value_code_tests::drop_that_panics_during_set::shared",
            || {
                let rt = Rc::new(Runtime::<Shared>::new());
                prime_reclaim_timer(&rt);
                reclaim_probe::force(RetireSite::SetRetire);
                super::drop_that_panics_during_set(rt);
            },
        )
        .assert_clean();
    }
}

mod every_engine_path_is_refused {
    use super::*;

    #[test]
    fn local() {
        every_path_is_refused(Rc::new(Runtime::<Local>::new()), engine_paths());
    }

    #[test]
    fn shared() {
        every_path_is_refused(Rc::new(Runtime::<Shared>::new()), engine_paths());
    }
}

mod every_collection_path_is_refused {
    use super::*;

    #[test]
    fn local() {
        every_path_is_refused(Rc::new(Runtime::<Local>::new()), collection_paths());
    }

    #[test]
    fn shared() {
        every_path_is_refused(Rc::new(Runtime::<Shared>::new()), collection_paths());
    }
}
