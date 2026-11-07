use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use super::collection::fp_of_elements;
use super::dep_index::DepFp;
use super::{Options, Store};
use crate::{Backend, Local, MemoryBackend};

/// A Python-style number: `Int(1)` and `Float(1.0)` are equal, so the
/// collection keeps whichever representative was inserted last.
#[derive(Clone, Debug, Serialize, Deserialize)]
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn value(&self) -> f64 {
        match self {
            Num::Int(i) => *i as f64,
            Num::Float(f) => *f,
        }
    }
}

impl PartialEq for Num {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Eq for Num {}

impl Hash for Num {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value().to_bits().hash(state);
    }
}

#[test]
fn restoring_a_swapped_representative_leaves_the_fingerprint_and_checkpoint_at_the_base() {
    let log = MemoryBackend::new();
    let store = Store::<Local, _>::open(log.clone(), Options::default()).unwrap();
    let c = store.collection::<Num>("c").unwrap();
    c.insert(&store, Num::Int(1)).unwrap();
    c.insert(&store, Num::Int(2)).unwrap();
    let deps = store.open_dep_index().unwrap();
    store.checkpoint().unwrap();
    let base = log.load_snapshot().unwrap().unwrap().1;
    let rt = store.runtime();
    let fp_now = || {
        let elements = c.handle().values(rt);
        fp_of_elements("c", &elements).unwrap()
    };
    let known = || match deps.fp("c") {
        Some(DepFp::Known(fp)) => fp,
        other => panic!("expected a known fingerprint, found {other:?}"),
    };
    assert_eq!(known(), fp_now());

    let scope = store.unlogged_scope().unwrap();
    let handle = scope.collection(&c).unwrap();
    handle.delete(rt, &Num::Int(1));
    handle.insert(rt, Num::Float(1.0));
    assert_ne!(known(), fp_now());
    handle.delete(rt, &Num::Float(1.0));
    handle.insert(rt, Num::Int(1));
    scope.close().unwrap();

    assert_eq!(known(), fp_now());
    store.checkpoint().unwrap();
    assert_eq!(log.load_snapshot().unwrap().unwrap().1, base);
}
