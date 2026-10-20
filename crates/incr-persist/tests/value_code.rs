//! The value-code contract through the store: a `Serialize`, `Hash` or
//! `Eq` of a persisted element may read Clean nodes and must not
//! otherwise call into the runtime or the store, and a compute must not
//! call into the store.

#[path = "../../incr-core/tests/common/child.rs"]
mod child;

use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

use child::run_in_child;
use incr_core::with_value_code;
use incr_persist::{Cells, Error, Local, MemoryBackend, Options, Shared, Store};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};

const CONTRACT_MESSAGE: &str = "value and key methods must not call into the runtime";
const FRAME_MESSAGE: &str = "Store methods must not be called from inside a query's compute";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Serialize,
    Hash,
    Compute,
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

/// Value code that calls the store panics before the store mutex is
/// requested, so the store is not poisoned.
fn store_call_from_value_code<C: Cells>() {
    let store = Store::<C, _>::open(MemoryBackend::new(), Options::default()).unwrap();
    let err = catch_unwind(AssertUnwindSafe(|| {
        with_value_code(|| store.restored_keys(""))
    }))
    .expect_err("the store call was allowed");
    let message = panic_message(err);
    assert!(
        message.contains(CONTRACT_MESSAGE),
        "expected the contract panic, got: {message}"
    );
    assert!(store.restored_keys("").unwrap().is_empty());
}

#[test]
fn local_store_call_from_value_code() {
    store_call_from_value_code::<Local>();
}

#[test]
fn shared_store_call_from_value_code() {
    store_call_from_value_code::<Shared>();
}

/// A compute that calls the store panics in release builds too, for a
/// method that would otherwise succeed.
fn store_call_from_a_compute<C: Cells>() {
    let store = Rc::new(Store::<C, _>::open(MemoryBackend::new(), Options::default()).unwrap());
    let q = store.runtime().create_query(|_| fire(Op::Compute));
    let hooked = Rc::clone(&store);
    arm(Op::Compute, move || {
        hooked.restored_keys("").unwrap();
    });
    let err = catch_unwind(AssertUnwindSafe(|| store.runtime().get(q)))
        .expect_err("the store call was allowed");
    let message = panic_message(err);
    assert!(
        message.contains(FRAME_MESSAGE),
        "expected the frame panic, got: {message}"
    );
    assert!(store.restored_keys("").unwrap().is_empty());
}

#[test]
fn local_store_call_from_a_compute() {
    store_call_from_a_compute::<Local>();
}

#[test]
fn shared_store_call_from_a_compute() {
    store_call_from_a_compute::<Shared>();
}

/// A hang test: the element's `Hash` runs under the store mutex, so a
/// `Store::input` from it would wait on that mutex forever. It panics
/// with the contract message instead, and the child exits.
fn insert_whose_element_hash_declares_an_input<C: Cells>() {
    let store = Rc::new(Store::<C, _>::open(MemoryBackend::new(), Options::default()).unwrap());
    let tags = store.collection::<Hooked>("tags").unwrap();
    let hooked = Rc::clone(&store);
    arm(Op::Hash, move || {
        hooked.input("other", 0u32).unwrap();
    });
    tags.insert(&store, Hooked(1)).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_insert_whose_element_hash_declares_an_input() {
    run_in_child("local_insert_whose_element_hash_declares_an_input", || {
        insert_whose_element_hash_declares_an_input::<Local>();
    })
    .assert_panicked_with(CONTRACT_MESSAGE);
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_insert_whose_element_hash_declares_an_input() {
    run_in_child("shared_insert_whose_element_hash_declares_an_input", || {
        insert_whose_element_hash_declares_an_input::<Shared>();
    })
    .assert_panicked_with(CONTRACT_MESSAGE);
}
