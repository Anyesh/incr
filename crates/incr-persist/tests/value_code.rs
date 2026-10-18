//! The value-code contract through the store: a `Serialize`, `Hash` or
//! `Eq` of a persisted element may read Clean nodes and must not
//! otherwise call into the runtime.

use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use incr_persist::{Cells, Error, Local, MemoryBackend, Options, Shared, Store};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

const CONTRACT_MESSAGE: &str = "value and key methods must not call into the runtime";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Serialize,
    Hash,
}

type Hook = (Op, Box<dyn Fn()>);

thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

fn arm(on: Op, f: impl Fn() + 'static) {
    HOOK.with(|h| *h.borrow_mut() = Some((on, Box::new(f))));
}

fn fire(op: Op) {
    let taken = HOOK.with(|h| {
        let mut h = h.borrow_mut();
        match &*h {
            Some((on, _)) if *on == op => h.take(),
            _ => None,
        }
    });
    if let Some((_, f)) = taken {
        f();
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Deserialize)]
struct Hooked(u32);

impl Serialize for Hooked {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        fire(Op::Serialize);
        serializer.serialize_u32(self.0)
    }
}

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

/// An insert whose element's `op` reads a Clean node succeeds, because
/// a Clean read takes no lock and never waits.
fn insert_whose_element_reads_a_clean_node<C: Cells>(op: Op) {
    let store = Rc::new(Store::<C, _>::open(MemoryBackend::new(), Options::default()).unwrap());
    let tags = store.collection::<Hooked>("tags").unwrap();
    let probe = store.input("probe", 5u32).unwrap();
    let hooked = Rc::clone(&store);
    arm(op, move || {
        assert_eq!(hooked.runtime().get(probe.handle()), 5);
    });
    tags.insert(&store, Hooked(1)).unwrap();
    assert!(HOOK.with(|h| h.borrow().is_none()), "the hook never ran");
    assert_eq!(tags.handle().values(store.runtime()), vec![Hooked(1)]);
}

/// An insert whose element's `op` reads a node that needs a compute
/// panics with the contract message, leaves the collection unchanged and
/// poisons the store, as any panic under its lock does.
fn insert_whose_element_reads_a_node_to_compute<C: Cells>(op: Op) {
    let store = Rc::new(Store::<C, _>::open(MemoryBackend::new(), Options::default()).unwrap());
    let tags = store.collection::<Hooked>("tags").unwrap();
    let probe = store.input("probe", 5u32).unwrap();
    let doubled = store
        .runtime()
        .create_query(move |rt| rt.get(probe.handle()) * 2);
    let hooked = Rc::clone(&store);
    arm(op, move || {
        hooked.runtime().get(doubled);
    });
    let err = catch_unwind(AssertUnwindSafe(|| tags.insert(&store, Hooked(1))))
        .expect_err("the runtime call was allowed");
    let message = panic_message(err);
    assert!(
        message.contains(CONTRACT_MESSAGE),
        "expected the contract panic, got: {message}"
    );
    assert!(tags.handle().values(store.runtime()).is_empty());
    assert!(matches!(
        tags.insert(&store, Hooked(2)),
        Err(Error::Poisoned)
    ));
}

fn insert_whose_element_reads_a_node<C: Cells>(op: Op) {
    insert_whose_element_reads_a_clean_node::<C>(op);
    insert_whose_element_reads_a_node_to_compute::<C>(op);
}

#[test]
fn local_serialize_that_reads_a_node() {
    insert_whose_element_reads_a_node::<Local>(Op::Serialize);
}

#[test]
fn shared_serialize_that_reads_a_node() {
    insert_whose_element_reads_a_node::<Shared>(Op::Serialize);
}

#[test]
fn local_hash_that_reads_a_node() {
    insert_whose_element_reads_a_node::<Local>(Op::Hash);
}

#[test]
fn shared_hash_that_reads_a_node() {
    insert_whose_element_reads_a_node::<Shared>(Op::Hash);
}
