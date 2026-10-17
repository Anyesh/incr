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
}

impl<C: Cells> Fixture<C> {
    fn new(rt: &Runtime<C>) -> Self {
        let clean = rt.create_input(5u32);
        let source = rt.create_input(1u32);
        let dirty = rt.create_query(move |rt| rt.get(source) + 1);
        let victim = rt.create_input(0u32);
        let observer = rt.observe(clean, |_| {});
        let c = rt.create_collection::<u32>();
        c.insert(rt, 1);
        rt.get(clean);
        rt.get(dirty);
        Self {
            clean,
            source,
            dirty,
            victim,
            observer,
            c,
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
    clone_that_reads_a_clean_node,
    clone_that_creates_a_node,
    partial_eq_of_a_recompute_that_reads_a_node,
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
