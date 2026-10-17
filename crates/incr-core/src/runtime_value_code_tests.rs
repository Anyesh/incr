//! The value-code contract: a user `Clone`, `PartialEq`, `Hash` or `Eq`
//! the engine calls may not call back into the runtime. Each case runs
//! under `Local` and `Shared`, through a hook that the chosen trait
//! method of `Hooked` fires on its n-th call.

use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use super::Runtime;
use crate::cells::{Cells, Local, Shared};
use crate::value_code::CONTRACT_MESSAGE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Clone,
    Eq,
    Hash,
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
    let taken = HOOK.with(|h| {
        let mut h = h.borrow_mut();
        match h.as_mut() {
            Some(hook) if hook.on == op => {
                hook.nth -= 1;
                (hook.nth == 0).then(|| h.take().unwrap())
            }
            _ => None,
        }
    });
    if let Some(mut hook) = taken {
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

fn partial_eq_that_deletes_the_input<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let hooked = Rc::clone(&rt);
    arm(Op::Eq, move || hooked.delete_node(a));
    assert_contract_panic("delete from PartialEq", || rt.set(a, Hooked(2)));
    assert!(rt.is_alive(a));
    assert_eq!(rt.get(a).0, 1);
    rt.set(a, Hooked(3));
    assert_eq!(rt.get(a).0, 3);
}

fn partial_eq_that_sets_another_input<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let b = rt.create_input(10u32);
    let hooked = Rc::clone(&rt);
    arm(Op::Eq, move || hooked.set(b, 11));
    assert_contract_panic("set from PartialEq", || rt.set(a, Hooked(2)));
    assert_eq!(rt.get(b), 10);
    assert_eq!(rt.get(a).0, 1);
    rt.set(a, Hooked(3));
    assert_eq!(rt.get(a).0, 3);
}

fn partial_eq_that_inserts_into_a_collection<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let c = rt.create_collection::<u32>();
    let hooked = Rc::clone(&rt);
    let target = c.clone();
    arm(Op::Eq, move || target.insert(&hooked, 7));
    assert_contract_panic("insert from PartialEq", || rt.set(a, Hooked(2)));
    assert_eq!(c.snapshot_len(), 0);
    assert_eq!(rt.get(a).0, 1);
    c.insert(&rt, 7);
    assert_eq!(c.values(&rt), vec![7]);
}

fn clone_that_reads_a_clean_node<C: Cells>(rt: Rc<Runtime<C>>) {
    let a = rt.create_input(Hooked(1));
    let b = rt.create_input(5u32);
    assert_eq!(rt.get(b), 5);
    let hooked = Rc::clone(&rt);
    arm(Op::Clone, move || {
        hooked.get(b);
    });
    assert_contract_panic("Clean get from Clone", || {
        rt.get(a);
    });
    assert_eq!(rt.get(a).0, 1);
    rt.set(a, Hooked(2));
    assert_eq!(rt.get(a).0, 2);
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

fn partial_eq_of_a_recompute_that_reads_a_node<C: Cells>(rt: Rc<Runtime<C>>) {
    let x = rt.create_input(1u32);
    let other = rt.create_input(5u32);
    let q = rt.create_query(move |rt| Hooked(rt.get(x)));
    assert_eq!(rt.get(q).0, 1);
    assert_eq!(rt.get(other), 5);
    rt.set(x, 2);
    let hooked = Rc::clone(&rt);
    arm(Op::Eq, move || {
        hooked.get(other);
    });
    assert_contract_panic("get from write_if_changed", || {
        rt.get(q);
    });
    rt.set(x, 3);
    assert_eq!(rt.get(q).0, 3);
}

/// The nodes and collections the checked-path table calls into, built
/// before the hook is armed.
struct Fixture<C: Cells> {
    clean: crate::Incr<u32>,
    source: crate::Incr<u32>,
    dirty: crate::Incr<u32>,
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
        ("get of a Clean node", |rt, f| {
            rt.get(f.clean);
        }),
        ("get of a Dirty node", |rt, f| {
            rt.get(f.dirty);
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
        hooked.get(b);
    });
    assert_contract_panic("get from Hash on insert", || c.insert(&rt, Hooked(1)));
    assert_eq!(c.snapshot_len(), 0);
    c.insert(&rt, Hooked(1));
    assert_eq!(ids(&rt, &c), vec![1]);
}

fn eq_that_reads_a_node_on_insert<C: Cells>(rt: Rc<Runtime<C>>) {
    let b = rt.create_input(5u32);
    let c = rt.create_collection::<Hooked>();
    c.insert(&rt, Hooked(1));
    let hooked = Rc::clone(&rt);
    arm(Op::Eq, move || {
        hooked.get(b);
    });
    assert_contract_panic("get from Eq on insert", || c.insert(&rt, Hooked(1)));
    assert_eq!(c.snapshot_len(), 1);
    c.insert(&rt, Hooked(1));
    assert_eq!(ids(&rt, &c), vec![1, 1]);
}

fn clone_that_reads_a_node_on_values<C: Cells>(rt: Rc<Runtime<C>>) {
    let b = rt.create_input(5u32);
    let c = rt.create_collection::<Hooked>();
    c.insert(&rt, Hooked(1));
    let hooked = Rc::clone(&rt);
    arm(Op::Clone, move || {
        hooked.get(b);
    });
    assert_contract_panic("get from Clone on values", || {
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
        hooked.get(b);
    });
    assert_contract_panic("get from Clone in a filter batch", || {
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
        hooked.get(b);
    });
    assert_contract_panic("get from Hash in the group_by key cache", || {
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
        hooked.get(b);
    });
    assert_contract_panic("get from Hash in the join key cache", || {
        rt.get(joined.version_node());
    });
    // The panic landed inside the batch, so the next run rebuilds: the
    // first two hashes file the sides, the third is the output converge.
    left.insert(&rt, Hooked(2));
    let hooked = Rc::clone(&rt);
    arm_nth(Op::Hash, 3, move || {
        hooked.get(b);
    });
    assert_contract_panic("get from Hash in a rebuild's converge", || {
        rt.get(joined.version_node());
    });
    right.insert(&rt, Hooked(2));
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
        hooked.get(b);
    });
    assert_contract_panic("get from Clone in the sorted view", || {
        rt.get(sorted.version_node());
    });
    c.insert(&rt, Hooked(1));
    rt.get(sorted.version_node());
    let hooked = Rc::clone(&rt);
    arm(Op::Clone, move || {
        hooked.get(b);
    });
    assert_contract_panic("get from Clone in a pairwise batch", || {
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
    clone_that_reads_a_clean_node,
    clone_that_creates_a_node,
    partial_eq_of_a_recompute_that_reads_a_node,
    hash_that_reads_a_node_on_insert,
    eq_that_reads_a_node_on_insert,
    clone_that_reads_a_node_on_values,
    clone_that_reads_a_node_in_a_filter_batch,
    hash_that_reads_a_node_in_a_group_by_batch,
    hash_that_reads_a_node_in_a_join_batch_and_its_rebuild,
    clone_that_reads_a_node_in_a_sort_batch,
);

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
