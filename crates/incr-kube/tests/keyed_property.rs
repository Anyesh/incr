//! Property tests for `KeyedCollection`, mirroring
//! `incr-core/tests/collection_property.rs`'s pattern: apply a random
//! sequence of keyed ops to both the real type and a `HashMap`-based
//! baseline model that applies the same staleness rule, then compare.
//!
//! `Payload` deliberately has no `Hash`/`Eq` impl (only `PartialEq`, via
//! the `f64` it wraps), to exercise the design's central claim that `V`
//! never needs those for `KeyedCollection<K, V>` to work.

use incr_concurrent::Runtime;
use incr_kube::KeyedCollection;
use proptest::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
struct Payload(f64);

#[derive(Clone, Debug)]
enum Op {
    Upsert(u64, u64, Payload),
    Remove(u64),
}

const KEY_SPACE: u64 = 8;

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0..KEY_SPACE, 0_u64..20, -50.0_f64..50.0).prop_map(|(k, v, x)| Op::Upsert(
            k,
            v,
            Payload(x)
        )),
        (0..KEY_SPACE).prop_map(Op::Remove),
    ]
}

/// Applies the same staleness rule `KeyedCollection::upsert` does:
/// `version <= stored.version` is dropped, not applied.
fn apply_to_baseline(ops: &[Op]) -> HashMap<u64, (u64, Payload)> {
    let mut baseline: HashMap<u64, (u64, Payload)> = HashMap::new();
    for op in ops {
        match op {
            Op::Upsert(k, v, p) => match baseline.get(k) {
                Some((stored_v, _)) if *stored_v >= *v => {}
                _ => {
                    baseline.insert(*k, (*v, p.clone()));
                }
            },
            Op::Remove(k) => {
                baseline.remove(k);
            }
        }
    }
    baseline
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn keyed_upsert_matches_baseline(ops in prop::collection::vec(op_strategy(), 0..60)) {
        let rt = Runtime::new();
        let kc: KeyedCollection<u64, Payload> = KeyedCollection::new(&rt);
        for op in &ops {
            match op {
                Op::Upsert(k, v, p) => {
                    kc.upsert(&rt, *k, *v, p.clone());
                }
                Op::Remove(k) => {
                    kc.remove(&rt, k);
                }
            }
        }

        let baseline = apply_to_baseline(&ops);

        prop_assert_eq!(kc.len(), baseline.len());
        for key in 0..KEY_SPACE {
            match baseline.get(&key) {
                Some((_, expected)) => {
                    let got = kc.get(&key);
                    prop_assert_eq!(got.as_deref(), Some(expected));
                }
                None => {
                    prop_assert!(kc.get(&key).is_none());
                }
            }
        }

        let count = kc.collection().count(&rt);
        prop_assert_eq!(rt.get(count) as usize, baseline.len());
    }
}
