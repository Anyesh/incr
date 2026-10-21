//! The mapped-write hooks incr-persist calls under its `DepIndex` hold
//! (T10): `set_returning_displaced` and the collection mutations with a
//! representative sink.

use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use proptest::prelude::*;

use super::{
    changed_at, collection_delete, collection_insert, collection_replace, set_returning_displaced,
    Displaced, RepSide, RepSink,
};
use crate::cells::{Cells, Local, Shared};
use crate::runtime::Runtime;
use crate::value_slot::Compared;

/// Compares by `id` only, and logs its `tag` when dropped, so a test can
/// tell which of two equal values was dropped and when.
#[derive(Clone, Debug)]
struct Tracked {
    id: u32,
    tag: u32,
    drops: Arc<Mutex<Vec<u32>>>,
}

impl PartialEq for Tracked {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Tracked {}

impl Hash for Tracked {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        self.drops.lock().unwrap().push(self.tag);
    }
}

struct Drops(Arc<Mutex<Vec<u32>>>);

impl Drops {
    fn new() -> Self {
        Self(Arc::default())
    }

    fn value(&self, id: u32, tag: u32) -> Tracked {
        Tracked {
            id,
            tag,
            drops: Arc::clone(&self.0),
        }
    }

    fn of(&self, tag: u32) -> usize {
        self.0.lock().unwrap().iter().filter(|&&t| t == tag).count()
    }
}

/// An element whose equality ignores `tag`, so the representative the
/// collection stores is observable.
#[derive(Clone, Copy, Debug)]
struct Item {
    key: u8,
    tag: u32,
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Item {}

impl Hash for Item {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

fn item(key: u8, tag: u32) -> Item {
    Item { key, tag }
}

#[derive(Default)]
struct Recorded(Vec<(RepSide, u8, u32)>);

impl RepSink<Item> for Recorded {
    fn on_rep(&mut self, side: RepSide, rep: &Item) {
        self.0.push((side, rep.key, rep.tag));
    }
}

fn set_hands_back_what_it_displaced<C: Cells>() {
    let rt = Runtime::<C>::new();
    let drops = Drops::new();
    let x = rt.create_input(drops.value(1, 10));
    let runs = Arc::new(AtomicUsize::new(0));
    let q = {
        let runs = Arc::clone(&runs);
        rt.create_query(move |rt| {
            runs.fetch_add(1, Ordering::Relaxed);
            rt.get(x).id
        })
    };
    assert_eq!(rt.get(q), 1);

    let before = drops.of(10);
    let out = set_returning_displaced(&rt, x, drops.value(2, 20)).0;
    assert!(matches!(out, Compared::Replaced(_)), "a change writes");
    assert_eq!(drops.of(10), before, "the displaced value is not dropped");
    drop(out);
    if C::SINGLE_THREADED {
        assert_eq!(drops.of(10), before + 1);
    }
    assert_eq!(rt.get(q), 2, "the walk ran");
    assert_eq!(runs.load(Ordering::Relaxed), 2);

    let out = set_returning_displaced(&rt, x, drops.value(2, 21)).0;
    match &out {
        Compared::Equal(rejected) => assert_eq!(rejected.tag, 21),
        Compared::Replaced(_) => panic!("an equal value must not write"),
    }
    assert_eq!(drops.of(21), 0, "the rejected value is not dropped");
    drop(out);
    assert_eq!(drops.of(21), 1);
    assert_eq!(rt.get(x).tag, 20, "a no-op keeps the earlier value");
    assert_eq!(rt.get(q), 2);
    assert_eq!(runs.load(Ordering::Relaxed), 2, "a no-op marks nothing");
}

#[test]
fn set_returning_displaced_local() {
    set_hands_back_what_it_displaced::<Local>();
}

#[test]
fn set_returning_displaced_shared() {
    set_hands_back_what_it_displaced::<Shared>();
}

#[test]
fn set_returning_displaced_hands_back_the_old_value_under_local() {
    let rt = Runtime::<Local>::new();
    let drops = Drops::new();
    let x = rt.create_input(drops.value(1, 10));
    let before = drops.of(10);
    match set_returning_displaced(&rt, x, drops.value(2, 20)).0 {
        Compared::Replaced(Some(old)) => assert_eq!(old.tag, 10),
        _ => panic!("a change hands back the old value"),
    }
    assert_eq!(drops.of(10), before + 1);
}

fn collection_sink_sees_each_side<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<Item>();
    let mut keep = Vec::new();
    let mut sink = Recorded::default();
    let mut step = |f: &mut dyn FnMut(&mut Recorded, &mut Vec<Displaced<Item>>)| {
        sink.0.clear();
        f(&mut sink, &mut keep);
        std::mem::take(&mut sink.0)
    };
    use RepSide::{Added, Removed};

    let got = step(&mut |s, k| collection_insert(&rt, &c, item(1, 10), s, k));
    assert_eq!(
        got,
        [(Added, 1, 10)],
        "an absent key reports the incoming value"
    );
    let got = step(&mut |s, k| collection_insert(&rt, &c, item(1, 11), s, k));
    assert_eq!(
        got,
        [(Added, 1, 10)],
        "a present key reports the stored one"
    );

    let got = step(&mut |s, k| {
        assert!(collection_delete(&rt, &c, &item(1, 12), s, k));
    });
    assert_eq!(got, [(Removed, 1, 10)], "a delete reports the stored one");
    let got = step(&mut |s, k| {
        assert!(!collection_delete(&rt, &c, &item(9, 0), s, k));
    });
    assert!(got.is_empty(), "a missing delete changes nothing");

    let got = step(&mut |s, k| {
        assert!(collection_replace(
            &rt,
            &c,
            Some(&item(1, 13)),
            item(1, 14),
            s,
            k
        ));
    });
    assert_eq!(
        got,
        [(Removed, 1, 10), (Added, 1, 14)],
        "a count-1 equal replace stores the new value"
    );
    assert_eq!(c.values(&rt)[0].tag, 14);

    step(&mut |s, k| collection_insert(&rt, &c, item(1, 15), s, k));
    let got = step(&mut |s, k| {
        assert!(collection_replace(
            &rt,
            &c,
            Some(&item(1, 16)),
            item(1, 17),
            s,
            k
        ));
    });
    assert_eq!(
        got,
        [(Removed, 1, 14), (Added, 1, 14)],
        "a count-2 equal replace keeps the stored value"
    );

    let got = step(&mut |s, k| {
        assert!(collection_replace(
            &rt,
            &c,
            Some(&item(1, 18)),
            item(2, 19),
            s,
            k
        ));
    });
    assert_eq!(got, [(Removed, 1, 14), (Added, 2, 19)]);
    let got = step(&mut |s, k| {
        assert!(!collection_replace(
            &rt,
            &c,
            Some(&item(7, 0)),
            item(3, 20),
            s,
            k
        ));
    });
    assert_eq!(got, [(Added, 3, 20)], "a missing old reports only the add");
    let got = step(&mut |s, k| {
        assert!(!collection_replace(&rt, &c, None, item(2, 21), s, k));
    });
    assert_eq!(got, [(Added, 2, 19)]);
    assert_eq!(
        rt.get(c.version_node()),
        12,
        "one version per recorded delta"
    );
}

#[test]
fn collection_sink_sees_each_side_local() {
    collection_sink_sees_each_side::<Local>();
}

#[test]
fn collection_sink_sees_each_side_shared() {
    collection_sink_sees_each_side::<Shared>();
}

/// Removes what the sink is told is removed and adds what it is told is
/// added, so it tracks the representatives the collection stores.
#[derive(Default)]
struct Model(Vec<(u8, u32)>);

impl RepSink<Item> for Model {
    fn on_rep(&mut self, side: RepSide, rep: &Item) {
        match side {
            RepSide::Added => self.0.push((rep.key, rep.tag)),
            RepSide::Removed => {
                let at = self
                    .0
                    .iter()
                    .position(|&e| e == (rep.key, rep.tag))
                    .expect("removed a representative the collection does not store");
                self.0.swap_remove(at);
            }
        }
    }
}

#[derive(Clone, Debug)]
enum Op {
    Insert(u8, u32),
    Delete(u8),
    Replace(Option<u8>, u8, u32),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0..4u8, any::<u32>()).prop_map(|(k, t)| Op::Insert(k, t)),
        (0..4u8).prop_map(Op::Delete),
        (proptest::option::of(0..4u8), 0..4u8, any::<u32>())
            .prop_map(|(o, k, t)| Op::Replace(o, k, t)),
    ]
}

fn sink_tracks_the_stored_representatives<C: Cells>(ops: &[Op]) {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<Item>();
    let mut model = Model::default();
    let mut keep = Vec::new();
    for op in ops {
        match *op {
            Op::Insert(k, t) => collection_insert(&rt, &c, item(k, t), &mut model, &mut keep),
            Op::Delete(k) => {
                collection_delete(&rt, &c, &item(k, 0), &mut model, &mut keep);
            }
            Op::Replace(old, k, t) => {
                let old = old.map(|o| item(o, 0));
                collection_replace(&rt, &c, old.as_ref(), item(k, t), &mut model, &mut keep);
            }
        }
        let mut stored: Vec<_> = c.values(&rt).iter().map(|e| (e.key, e.tag)).collect();
        stored.sort_unstable();
        let mut tracked = model.0.clone();
        tracked.sort_unstable();
        assert_eq!(stored, tracked, "after {op:?}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn sink_tracks_the_stored_representatives_local(ops in proptest::collection::vec(op(), 0..60)) {
        sink_tracks_the_stored_representatives::<Local>(&ops);
    }

    #[test]
    fn sink_tracks_the_stored_representatives_shared(ops in proptest::collection::vec(op(), 0..60)) {
        sink_tracks_the_stored_representatives::<Shared>(&ops);
    }
}

#[test]
fn collection_hooks_keep_displaced_elements_for_the_caller() {
    let rt = Runtime::<Local>::new();
    let drops = Drops::new();
    let c = rt.create_collection::<Tracked>();
    let mut keep = Vec::new();
    collection_insert(&rt, &c, drops.value(1, 10), &mut (), &mut keep);
    collection_insert(&rt, &c, drops.value(1, 11), &mut (), &mut keep);
    assert_eq!(drops.of(11), 0, "the unused incoming copy is kept");
    collection_delete(&rt, &c, &drops.value(1, 12), &mut (), &mut keep);
    collection_delete(&rt, &c, &drops.value(1, 13), &mut (), &mut keep);
    assert_eq!(drops.of(10), 0, "the removed stored value is kept");
    assert_eq!(c.snapshot_len(), 0);
    drop(keep);
    assert_eq!(drops.of(11), 1);
    assert_eq!(drops.of(10), 1);
}

struct CallsIn<'a, C: Cells> {
    rt: &'a Runtime<C>,
    other: crate::handle::Incr<u64>,
}

impl<C: Cells> RepSink<Item> for CallsIn<'_, C> {
    fn on_rep(&mut self, _: RepSide, _: &Item) {
        self.rt.set(self.other, 1);
    }
}

fn sink_runs_as_value_code_before_any_change<C: Cells>() {
    let rt = Runtime::<C>::new();
    let c = rt.create_collection::<Item>();
    let other = rt.create_input(0_u64);
    let mut keep = Vec::new();
    collection_insert(&rt, &c, item(1, 10), &mut (), &mut keep);
    let before = rt.get(c.version_node());
    let mut sink = CallsIn { rt: &rt, other };
    type Mutation<'m, 'a, C> = &'m mut dyn FnMut(&mut CallsIn<'a, C>, &mut Vec<Displaced<Item>>);
    let mutations: [Mutation<'_, '_, C>; 3] = [
        &mut |s, k| collection_insert(&rt, &c, item(2, 20), s, k),
        &mut |s, k| {
            collection_delete(&rt, &c, &item(1, 0), s, k);
        },
        &mut |s, k| {
            collection_replace(&rt, &c, Some(&item(1, 0)), item(3, 30), s, k);
        },
    ];
    for mutate in mutations {
        let err = catch_unwind(AssertUnwindSafe(|| mutate(&mut sink, &mut keep)))
            .expect_err("a sink that calls into the runtime panics");
        let msg = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert!(
            msg.contains(crate::value_code::CONTRACT_MESSAGE),
            "unexpected panic: {msg}"
        );
        let stored: Vec<_> = c.values(&rt).iter().map(|e| (e.key, e.tag)).collect();
        assert_eq!(stored, [(1, 10)], "the collection is untouched");
        assert_eq!(rt.get(c.version_node()), before);
        assert_eq!(rt.get(other), 0);
    }
}

#[test]
fn sink_runs_as_value_code_before_any_change_local() {
    sink_runs_as_value_code_before_any_change::<Local>();
}

#[test]
fn sink_runs_as_value_code_before_any_change_shared() {
    sink_runs_as_value_code_before_any_change::<Shared>();
}

fn changed_at_tracks_writes_that_wrote<C: Cells>() {
    let rt = Runtime::<C>::new();
    let x = rt.create_input(1_u64);
    let c0 = changed_at(&rt, x);
    rt.set(x, 2);
    let c1 = changed_at(&rt, x);
    assert!(c1 > c0, "a write that wrote moves the stamp");
    rt.set(x, 2);
    assert_eq!(changed_at(&rt, x), c1, "a no-op leaves it");
    assert!(matches!(
        set_returning_displaced(&rt, x, 3).0,
        Compared::Replaced(_)
    ));
    let c2 = changed_at(&rt, x);
    assert!(c2 > c1);
    assert!(matches!(
        set_returning_displaced(&rt, x, 3).0,
        Compared::Equal(_)
    ));
    assert_eq!(changed_at(&rt, x), c2);
}

#[test]
fn changed_at_tracks_writes_that_wrote_local() {
    changed_at_tracks_writes_that_wrote::<Local>();
}

#[test]
fn changed_at_tracks_writes_that_wrote_shared() {
    changed_at_tracks_writes_that_wrote::<Shared>();
}

#[test]
fn changed_at_refuses_a_stale_or_foreign_handle() {
    let rt = Runtime::<Shared>::new();
    let x = rt.create_input(1_u64);
    let other = Runtime::<Shared>::new();
    let err = catch_unwind(AssertUnwindSafe(|| changed_at(&other, x)))
        .expect_err("a foreign handle panics");
    drop(err);
    rt.delete_node(x);
    let err =
        catch_unwind(AssertUnwindSafe(|| changed_at(&rt, x))).expect_err("a stale handle panics");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or_default();
    assert!(msg.contains("stale"), "unexpected panic: {msg}");
}
