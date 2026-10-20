use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use incr_core::persist_hooks::changed_at;
use serde::{Deserialize, Serialize};

use super::binding::encode;
use super::dep_index::{fp_of, DepFp, DepIndex};
use super::{Options, Store};
use crate::{Cells, Local, MemoryBackend, Shared};

fn open<C: Cells>() -> Store<C, MemoryBackend> {
    Store::open(MemoryBackend::new(), Options::default()).unwrap()
}

fn h<T: Serialize>(value: &T) -> u128 {
    fp_of(&encode(value).unwrap())
}

fn known(deps: &DepIndex, key: &str) -> u128 {
    match deps.fp(key) {
        Some(DepFp::Known(fp)) => fp,
        other => panic!("{key}: expected a known fingerprint, found {other:?}"),
    }
}

#[test]
fn an_input_starts_at_the_hash_of_the_bytes_it_decoded() {
    let store = open::<Local>();
    store.input("x", 5u32).unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_eq!(known(&deps, "x"), h(&5u32));
}

#[test]
fn an_input_declared_over_a_logged_value_starts_at_its_bytes() {
    let backend = MemoryBackend::new();
    {
        let store = Store::<Local, _>::open(backend.clone(), Options::default()).unwrap();
        let x = store.input("x", 1u32).unwrap();
        store.set(x, 8).unwrap();
    }
    let store = Store::<Local, _>::open(backend, Options::default()).unwrap();
    let deps = store.open_dep_index().unwrap();
    store.input("x", 1u32).unwrap();
    assert_eq!(known(&deps, "x"), h(&8u32));
}

#[test]
fn mapped_writes_keep_the_input_fingerprint_without_a_memo() {
    let store = open::<Local>();
    let x = store.input("x", 1u32).unwrap();
    store.set(x, 2).unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_eq!(known(&deps, "x"), h(&2u32));
}

fn mapped_writes_update_the_index<C: Cells>() {
    let store = open::<C>();
    let x = store.input("x", 1u32).unwrap();
    let deps = store.open_dep_index().unwrap();
    store.set(x, 3).unwrap();
    assert_eq!(known(&deps, "x"), h(&3u32));
}

#[test]
fn mapped_writes_update_the_index_local() {
    mapped_writes_update_the_index::<Local>();
}

#[test]
fn mapped_writes_update_the_index_shared() {
    mapped_writes_update_the_index::<Shared>();
}

#[test]
fn an_equal_write_leaves_the_fingerprint_of_the_value_the_runtime_holds() {
    let store = open::<Local>();
    let x = store.input("x", 0.0f64).unwrap();
    store.set(x, -0.0).unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_eq!(known(&deps, "x"), h(&0.0f64));
    store.set(x, -0.0).unwrap();
    assert_eq!(known(&deps, "x"), h(&0.0f64));
    assert_ne!(h(&0.0f64), h(&-0.0f64));
}

#[test]
fn a_delete_resets_the_fingerprint_to_the_logged_default() {
    let store = open::<Local>();
    let x = store.input("x", 4u32).unwrap();
    let deps = store.open_dep_index().unwrap();
    store.set(x, 9).unwrap();
    store.delete(x).unwrap();
    assert_eq!(known(&deps, "x"), h(&4u32));
}

#[test]
fn a_collection_fingerprint_sums_its_stored_occurrences() {
    let store = open::<Local>();
    let c = store.collection::<u32>("c").unwrap();
    c.insert(&store, 1).unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_eq!(known(&deps, "c"), h(&1u32));
    c.insert(&store, 2).unwrap();
    c.insert(&store, 2).unwrap();
    let sum = h(&1u32).wrapping_add(h(&2u32)).wrapping_add(h(&2u32));
    assert_eq!(known(&deps, "c"), sum);
    c.delete(&store, &2).unwrap();
    assert_eq!(known(&deps, "c"), h(&1u32).wrapping_add(h(&2u32)));
    c.replace(&store, Some(&1), 3).unwrap();
    assert_eq!(known(&deps, "c"), h(&3u32).wrapping_add(h(&2u32)));
    store.delete(&c).unwrap();
    assert_eq!(known(&deps, "c"), 0);
}

/// Equal across signs, as `f64` is, so a replace of 0.0 by -0.0 is equal.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Num(f64);

impl PartialEq for Num {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Num {}

impl Hash for Num {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0 + 0.0).to_bits().hash(state);
    }
}

#[test]
fn an_equal_replace_of_a_single_occurrence_stores_the_new_encoding() {
    let store = open::<Local>();
    let c = store.collection::<Num>("c").unwrap();
    c.insert(&store, Num(0.0)).unwrap();
    let deps = store.open_dep_index().unwrap();
    c.replace(&store, Some(&Num(0.0)), Num(-0.0)).unwrap();
    assert_eq!(known(&deps, "c"), h(&Num(-0.0)));
}

#[test]
fn a_collection_declared_after_the_index_opens_is_registered_converged() {
    let store = open::<Local>();
    let c = store.collection::<u32>("c").unwrap();
    c.insert(&store, 6).unwrap();
    c.insert(&store, 7).unwrap();
    store.undeclare(&c).unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_eq!(deps.fp("c"), None);
    let c = store.collection::<u32>("c").unwrap();
    assert_eq!(known(&deps, "c"), h(&6u32).wrapping_add(h(&7u32)));
    assert_eq!(
        deps.key_of(c.handle().version_node().erase()).as_deref(),
        Some("c")
    );
}

#[test]
fn undeclare_unregisters_the_key() {
    let store = open::<Local>();
    let x = store.input("x", 1u32).unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_eq!(deps.key_of(x.handle().erase()).as_deref(), Some("x"));
    store.undeclare(x).unwrap();
    assert_eq!(deps.fp("x"), None);
    assert_eq!(deps.key_of(x.handle().erase()), None);
}

#[test]
fn an_undeclare_that_fails_keeps_the_key_registered() {
    let store = open::<Local>();
    let x = store.input("x", 1u32).unwrap();
    let handle = x.handle();
    let reader = store.runtime().create_query(move |rt| rt.get(handle) + 1);
    assert_eq!(store.runtime().get(reader), 2);
    let deps = store.open_dep_index().unwrap();
    assert!(store.undeclare(x).is_err());
    assert_eq!(known(&deps, "x"), h(&1u32));
}

#[test]
fn opening_the_index_twice_shares_it() {
    let store = open::<Local>();
    let a = store.open_dep_index().unwrap();
    let b = store.open_dep_index().unwrap();
    assert!(Arc::ptr_eq(&a, &b));
}

/// Panics when compared with 13.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Bomb(u32);

impl Hash for Bomb {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for Bomb {
    fn eq(&self, other: &Self) -> bool {
        assert!(self.0 != 13 && other.0 != 13, "boom");
        self.0 == other.0
    }
}

impl Eq for Bomb {}

#[test]
fn a_panic_in_a_mapped_input_write_leaves_the_key_untrusted() {
    let store = open::<Local>();
    let x = store.input("x", Bomb(1)).unwrap();
    let deps = store.open_dep_index().unwrap();
    let outcome = catch_unwind(AssertUnwindSafe(|| store.set(x, Bomb(13))));
    assert!(outcome.is_err());
    assert_eq!(deps.fp("x"), Some(DepFp::Untrusted));
    assert_eq!(deps.stamp("x"), None, "an untrusted key keeps no stamp");
}

/// Panics when hashed with 13.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct HashBomb(u32);

impl Hash for HashBomb {
    fn hash<H: Hasher>(&self, state: &mut H) {
        assert!(self.0 != 13, "boom");
        self.0.hash(state);
    }
}

#[test]
fn a_panic_in_a_mapped_collection_write_leaves_the_key_untrusted() {
    let store = open::<Local>();
    let c = store.collection::<HashBomb>("c").unwrap();
    c.insert(&store, HashBomb(1)).unwrap();
    let deps = store.open_dep_index().unwrap();
    let outcome = catch_unwind(AssertUnwindSafe(|| c.insert(&store, HashBomb(13))));
    assert!(outcome.is_err());
    assert_eq!(deps.fp("c"), Some(DepFp::Untrusted));
}

thread_local! {
    static WATCHED: RefCell<Option<Arc<DepIndex>>> = const { RefCell::new(None) };
    static DROPPED_UNDER_HOLD: RefCell<u32> = const { RefCell::new(0) };
}

/// Counts drops that run while the watched index is held for writing.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Probe(u32);

impl Drop for Probe {
    fn drop(&mut self) {
        WATCHED.with(|watched| {
            if let Some(deps) = watched.borrow().as_ref() {
                if deps.is_held() {
                    DROPPED_UNDER_HOLD.with(|n| *n.borrow_mut() += 1);
                }
            }
        });
    }
}

#[test]
fn displaced_values_are_dropped_after_the_hold() {
    let store = open::<Local>();
    let x = store.input("x", Probe(0)).unwrap();
    let c = store.collection::<Probe>("c").unwrap();
    c.insert(&store, Probe(1)).unwrap();
    let deps = store.open_dep_index().unwrap();
    WATCHED.with(|watched| *watched.borrow_mut() = Some(Arc::clone(&deps)));
    store.set(x, Probe(1)).unwrap();
    store.set(x, Probe(1)).unwrap();
    store.delete(x).unwrap();
    c.replace(&store, Some(&Probe(1)), Probe(2)).unwrap();
    c.delete(&store, &Probe(2)).unwrap();
    store.delete(&c).unwrap();
    WATCHED.with(|watched| *watched.borrow_mut() = None);
    assert_eq!(DROPPED_UNDER_HOLD.with(|n| *n.borrow()), 0);
}

#[test]
fn a_catch_up_across_a_compaction_keeps_the_fingerprints() {
    let backend = MemoryBackend::new();
    let a = Store::<Local, _>::open(backend.clone(), Options::default()).unwrap();
    a.input("x", 1u32).unwrap();
    a.collection::<u32>("c").unwrap();
    let deps = a.open_dep_index().unwrap();
    let b = Store::<Local, _>::open(backend, Options::default()).unwrap();
    let x = b.input("x", 1u32).unwrap();
    b.set(x, 5).unwrap();
    let c = b.collection::<u32>("c").unwrap();
    c.insert(&b, 3).unwrap();
    c.insert(&b, 4).unwrap();
    b.checkpoint().unwrap();
    a.refresh().unwrap();
    assert_eq!(known(&deps, "x"), h(&5u32));
    assert_eq!(known(&deps, "c"), h(&3u32).wrapping_add(h(&4u32)));
}

#[test]
fn a_collection_whose_node_was_deleted_registers_untrusted() {
    let store = open::<Local>();
    let c = store.collection::<u32>("c").unwrap();
    c.insert(&store, 1).unwrap();
    store
        .runtime()
        .try_delete_node(c.handle().version_node())
        .unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_eq!(deps.fp("c"), Some(DepFp::Untrusted));
}

fn stamp(deps: &DepIndex, key: &str) -> u64 {
    deps.stamp(key)
        .unwrap_or_else(|| panic!("{key}: expected a mapped stamp"))
}

#[test]
fn an_input_records_its_stamp_at_creation() {
    let store = open::<Local>();
    let x = store.input("x", 5u32).unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_eq!(stamp(&deps, "x"), changed_at(store.runtime(), x.handle()));
}

#[test]
fn an_input_created_while_the_index_is_open_registers_its_stamp() {
    let store = open::<Local>();
    let deps = store.open_dep_index().unwrap();
    let x = store.input("x", 5u32).unwrap();
    assert_eq!(stamp(&deps, "x"), changed_at(store.runtime(), x.handle()));
}

fn writes_that_wrote_move_the_stamp<C: Cells>() {
    let store = open::<C>();
    let x = store.input("x", 1u32).unwrap();
    store.set(x, 2).unwrap();
    let deps = store.open_dep_index().unwrap();
    let before = stamp(&deps, "x");
    assert_eq!(
        before,
        changed_at(store.runtime(), x.handle()),
        "kept without a memo"
    );
    store.set(x, 3).unwrap();
    let after = stamp(&deps, "x");
    assert!(after > before);
    assert_eq!(after, changed_at(store.runtime(), x.handle()));
    store.set(x, 3).unwrap();
    assert_eq!(stamp(&deps, "x"), after, "a no-op leaves it");
}

#[test]
fn writes_that_wrote_move_the_stamp_local() {
    writes_that_wrote_move_the_stamp::<Local>();
}

#[test]
fn writes_that_wrote_move_the_stamp_shared() {
    writes_that_wrote_move_the_stamp::<Shared>();
}

#[test]
fn a_no_op_mapped_write_cannot_launder_a_bypass() {
    let backend = MemoryBackend::new();
    let a = Store::<Local, _>::open(backend.clone(), Options::default()).unwrap();
    let b = Store::<Local, _>::open(backend, Options::default()).unwrap();
    let xa = a.input("x", 1u32).unwrap();
    let xb = b.input("x", 1u32).unwrap();
    let deps = a.open_dep_index().unwrap();
    let mapped = stamp(&deps, "x");
    a.runtime().set(xa.handle(), 9);
    b.set(xb, 9).unwrap();
    // Equal to the bypassed value, so the runtime treats it as a no-op.
    a.refresh().unwrap();
    assert_eq!(stamp(&deps, "x"), mapped);
    assert_ne!(changed_at(a.runtime(), xa.handle()), mapped);
}

#[test]
fn a_collection_has_no_stamp() {
    let store = open::<Local>();
    store.collection::<u32>("c").unwrap();
    let deps = store.open_dep_index().unwrap();
    assert_eq!(deps.stamp("c"), None);
}

#[test]
fn an_undeclare_that_fails_keeps_the_stamp() {
    let store = open::<Local>();
    let x = store.input("x", 1u32).unwrap();
    let deps = store.open_dep_index().unwrap();
    let before = stamp(&deps, "x");
    let h = x.handle();
    let q = store.runtime().create_query(move |rt| rt.get(h) + 1);
    store.runtime().get(q);
    assert!(store.undeclare(x).is_err());
    assert_eq!(stamp(&deps, "x"), before);
}
