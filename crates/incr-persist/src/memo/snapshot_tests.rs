use std::sync::Arc;

use super::KEYS_TOUCHED;
use crate::memo::{BuildId, MemoBuilder, MemoOptions, MemoryMemo};
use crate::{MemoryBackend, Options, Shared, Store};

/// `top(1) = leaf(1) + y` over 200 stored leaves `leaf(k) = k`. After `y`
/// moves only `top(1)` is dirty, with one clean dep: the checkpoint reads
/// those two keys, not every live entry's.
#[test]
fn a_checkpoint_touches_only_the_keys_of_dirty_entries_and_their_clean_deps() {
    let store =
        Arc::new(Store::<Shared, _>::open(MemoryBackend::new(), Options::default()).unwrap());
    let y = store.input("y", 10_u64).unwrap();
    let yh = y.handle();
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(0x5a)));
    let leaf = builder.table::<u32, u64>("leaf", 1);
    let top = builder.table::<u32, u64>("top", 1);
    builder.define(leaf, |_, &k| u64::from(k));
    builder.define(top, move |ctx, &k| ctx.get(leaf, k) + ctx.rt().get(yh));
    let memo = builder.open(&store, MemoryMemo::new()).unwrap();
    let rt = memo.runtime();
    let q = memo.get_or_insert(top, 1);
    assert_eq!(rt.get(q), 11);
    for k in 2..=200 {
        rt.get(memo.get_or_insert(leaf, k));
    }
    assert_eq!(memo.checkpoint().unwrap().written, 201);
    store.set(y, 20).unwrap();
    assert_eq!(rt.get(q), 21);
    KEYS_TOUCHED.with(|n| n.set(0));
    assert_eq!(memo.checkpoint().unwrap().written, 201);
    assert_eq!(KEYS_TOUCHED.with(std::cell::Cell::get), 2);
}
