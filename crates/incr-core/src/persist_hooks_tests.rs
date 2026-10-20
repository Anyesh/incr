//! The mapped-write hooks incr-persist calls under its `DepIndex` hold
//! (T10).

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::set_returning_displaced;
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
    let out = set_returning_displaced(&rt, x, drops.value(2, 20));
    assert!(matches!(out, Compared::Replaced(_)), "a change writes");
    assert_eq!(drops.of(10), before, "the displaced value is not dropped");
    drop(out);
    if C::SINGLE_THREADED {
        assert_eq!(drops.of(10), before + 1);
    }
    assert_eq!(rt.get(q), 2, "the walk ran");
    assert_eq!(runs.load(Ordering::Relaxed), 2);

    let out = set_returning_displaced(&rt, x, drops.value(2, 21));
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
    match set_returning_displaced(&rt, x, drops.value(2, 20)) {
        Compared::Replaced(Some(old)) => assert_eq!(old.tag, 10),
        _ => panic!("a change hands back the old value"),
    }
    assert_eq!(drops.of(10), before + 1);
}
