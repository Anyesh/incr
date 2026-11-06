//! `group_by` routing after a store or memo write: it runs once the write
//! has released its locks, so a `key_fn` that reads the source routes
//! without deadlock and group consumers are current with no router read.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use incr_core::{GroupedCollection, Incr};
use incr_persist::memo::{BuildId, MemoBuilder, MemoCells, MemoOptions, MemoryMemo};
use incr_persist::{Cells, Local, MemoryBackend, Options, Shared, Store};

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

both!(
    a_key_fn_reading_the_source_routes_after_a_store_write,
    a_key_fn_reading_an_external_routes_after_set_external,
);

thread_local! {
    static STASH: RefCell<Vec<Rc<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

/// Keeps `value` for a `key_fn` on this thread, which cannot capture a
/// `Local` store.
fn stash<T: 'static>(value: T) -> usize {
    STASH.with(|s| {
        let mut s = s.borrow_mut();
        s.push(Rc::new(value));
        s.len() - 1
    })
}

fn stashed<T: 'static>(at: usize) -> Rc<T> {
    let any = STASH.with(|s| Rc::clone(&s.borrow()[at]));
    any.downcast::<T>()
        .unwrap_or_else(|_| panic!("stash entry {at} has another type"))
}

fn group_len<C: Cells>(g: &GroupedCollection<u32, u32, C>, key: u32) -> Option<usize> {
    g.get_group(&key).map(|group| group.snapshot_len())
}

fn a_key_fn_reading_the_source_routes_after_a_store_write<C: Cells>() {
    let at = stash(Store::<C, _>::open(MemoryBackend::new(), Options::default()).unwrap());
    let store = stashed::<Store<C, MemoryBackend>>(at);
    let rt = store.runtime();
    let items = store.collection::<u32>("items").unwrap();
    let flip = store.input("flip", 0_u32).unwrap();
    let size = items.handle().count(rt);
    let flip_node = flip.handle();
    let g = items.handle().group_by(rt, move |x| {
        let store = stashed::<Store<C, MemoryBackend>>(at);
        let rt = store.runtime();
        x % 2 + 10 * u32::from(rt.get(size) > 2) + 100 * rt.get(flip_node)
    });
    items.insert(&store, 1).unwrap();
    let odd = g.get_group(&1).expect("the first insert's group").count(rt);
    assert_eq!(rt.get(odd), 1);

    items.insert(&store, 3).unwrap();
    assert_eq!(rt.get(odd), 2, "a group's count after a collection insert");
    assert!(rt.take_routing_failure().is_none());

    items.insert(&store, 4).unwrap();
    assert_eq!(rt.get(odd), 2);
    assert_eq!(
        group_len(&g, 10),
        Some(1),
        "a new row keyed by the source's size"
    );

    store.set(flip, 1).unwrap();
    items.insert(&store, 5).unwrap();
    assert_eq!(
        group_len(&g, 111),
        Some(1),
        "a new row keyed by the input set"
    );
    assert!(rt.take_routing_failure().is_none());
}

fn a_key_fn_reading_an_external_routes_after_set_external<C: MemoCells>() {
    let store = Arc::new(Store::<C, _>::open(MemoryBackend::new(), Options::default()).unwrap());
    let builder = MemoBuilder::<C>::new(MemoOptions::new(BuildId::Fixed(0xb1d)));
    let memo = builder.open(&store, MemoryMemo::new()).unwrap();
    let rt = memo.runtime();
    let cfg: Incr<u128> = memo.set_external("cfg", 0);
    let at = stash(Arc::clone(&store));
    let items = rt.create_collection::<u32>();
    let g = items.group_by(rt, move |x| {
        let store = stashed::<Arc<Store<C, MemoryBackend>>>(at);
        x % 2 + 10 * u32::from(store.runtime().get(cfg) > 0)
    });
    items.insert(rt, 1);
    let odd = g.get_group(&1).expect("the insert's group").count(rt);
    assert_eq!(rt.get(odd), 1);

    let queued = rt.hold_routing_suppressed();
    items.insert(rt, 3);
    drop(queued);
    memo.set_external("cfg", 7);
    assert_eq!(
        group_len(&g, 11),
        Some(1),
        "routed when set_external returns"
    );
    assert_eq!(rt.get(odd), 1);
    assert!(rt.take_routing_failure().is_none());
}
