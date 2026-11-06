//! `group_by` over a `KeyedCollection` routes once a write has released the
//! index lock, so a `key_fn` that reads the index does not deadlock and
//! group consumers are current with no read of the router.

use std::sync::Arc;

use incr_concurrent::{GroupedCollection, Runtime};
use incr_kube::{Entry, KeyedCollection};

type Keyed = KeyedCollection<String, i64>;

fn grouped_by_indexed_sign(
    rt: &Runtime,
    keyed: &Arc<Keyed>,
) -> GroupedCollection<i64, Entry<String, i64>> {
    let index = Arc::clone(keyed);
    keyed
        .collection()
        .group_by(rt, move |entry: &Entry<String, i64>| {
            index
                .get(entry.key())
                .expect("a routed entry's key is indexed")
                .signum()
        })
}

fn group_len<T: Clone + Eq + std::hash::Hash + Send + Sync + 'static>(
    g: &GroupedCollection<i64, T>,
    key: i64,
) -> usize {
    g.get_group(&key).map_or(0, |group| group.snapshot_len())
}

#[test]
fn a_key_fn_reading_the_index_routes_after_each_write() {
    let rt = Runtime::new();
    let keyed = Arc::new(Keyed::new(&rt));
    let g = grouped_by_indexed_sign(&rt, &keyed);

    keyed.upsert(&rt, "a".into(), 1, 5);
    let positive = g.get_group(&1).expect("routed by the upsert").count(&rt);
    assert_eq!(rt.get(positive), 1);

    keyed.upsert(&rt, "b".into(), 1, 7);
    assert_eq!(rt.get(positive), 2, "a group's count after an upsert");
    keyed.upsert(&rt, "a".into(), 2, -3);
    assert_eq!(rt.get(positive), 1, "after a replacing upsert");
    assert_eq!(group_len(&g, -1), 1, "a new key's group after an upsert");

    assert!(keyed.remove(&rt, &"b".to_string()));
    assert_eq!(rt.get(positive), 0, "after a remove");

    keyed.upsert(&rt, "c".into(), 1, 2);
    assert_eq!(keyed.retain(&rt, |key| key != "c"), 1);
    assert_eq!(rt.get(positive), 0, "after a retain");
    assert_eq!(group_len(&g, -1), 1);
    assert!(rt.take_routing_failure().is_none());
}

#[cfg(debug_assertions)]
mod lock_order {
    use std::hash::{Hash, Hasher};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, Ordering};

    use incr_concurrent::IncrCollection;

    use super::*;

    /// A key whose hash, once armed, inserts into a collection of another
    /// runtime, which the index's routing hold does not cover. Only the
    /// index lookup of a key already present hashes it outside value code.
    #[derive(Clone)]
    struct WritingKey {
        other: Arc<(Runtime, IncrCollection<i64>)>,
        armed: Arc<AtomicBool>,
    }

    impl PartialEq for WritingKey {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }

    impl Eq for WritingKey {}

    impl Hash for WritingKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            if self.armed.swap(false, Ordering::SeqCst) {
                let (rt, c) = &*self.other;
                c.insert(rt, 1);
            }
            0_u8.hash(state);
        }
    }

    fn upsert_writing_to_another_runtime(hold_other: bool) -> std::thread::Result<usize> {
        let other_rt = Runtime::new();
        let other_c = other_rt.create_collection::<i64>();
        let g = other_c.group_by(&other_rt, |x| x % 2);
        let other = Arc::new((other_rt, other_c));
        let key = WritingKey {
            other: Arc::clone(&other),
            armed: Arc::new(AtomicBool::new(false)),
        };
        let rt = Runtime::new();
        let keyed = KeyedCollection::<WritingKey, i64>::new(&rt);
        keyed.upsert(&rt, key.clone(), 1, 0);
        key.armed.store(true, Ordering::SeqCst);
        catch_unwind(AssertUnwindSafe(|| {
            {
                let _hold = hold_other.then(|| other.0.hold_routing());
                keyed.upsert(&rt, key, 2, 0);
            }
            group_len(&g, 1)
        }))
    }

    #[test]
    fn a_write_to_another_runtime_under_the_index_lock_panics_without_its_hold() {
        let payload =
            upsert_writing_to_another_runtime(false).expect_err("routing ran under the index lock");
        let text = payload
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert!(
            text.contains("routing runs while holding [Binding]"),
            "{text}"
        );
    }

    #[test]
    fn a_hold_taken_before_the_upsert_routes_the_other_runtime_after_it() {
        let routed = upsert_writing_to_another_runtime(true).expect("no lock-order panic");
        assert_eq!(routed, 1);
    }
}
