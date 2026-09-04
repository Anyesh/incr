//! Concurrent stress over `KeyedCollection`, mirroring
//! `incr-core/tests/stress.rs`'s pattern (writer thread(s), reader
//! threads pulling a derived view, a shared stop flag, iteration counts
//! scaled down under Miri).
//!
//! Split into two tests because a reader can't tell a legitimate
//! `remove` apart from a same-key replace's (now-fixed) transient by
//! looking at `count` alone, so one invariant can't cover both:
//! `replace_only` gives readers an exact constant to check on every
//! read (the test that would have caught the `replace`/`group_by`
//! transient-count bugs); `mixed_upsert_remove` only has bounds to
//! check.

use incr_concurrent::Runtime;
use incr_kube::KeyedCollection;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn iters(full: usize) -> usize {
    if cfg!(miri) {
        full / 50
    } else {
        full
    }
}

const KEY_SPACE: u64 = 64;

#[test]
fn replace_only() {
    let rt: Arc<Runtime> = Arc::new(Runtime::new());
    let kc: Arc<KeyedCollection<u64, u64>> = Arc::new(KeyedCollection::new(&rt));
    for key in 0..KEY_SPACE {
        kc.upsert(&rt, key, 1, key);
    }

    // Same key_fn on every replace means old and new always land in
    // the same group (the key itself never changes), the exact case
    // the group_by batching fix targets.
    let groups = kc.collection().group_by(&rt, |e| e.key() % 2);
    let _ = rt.get(groups.version_node());
    let evens = groups.get_group(&0).expect("even group missing");
    let odds = groups.get_group(&1).expect("odd group missing");
    let evens_count = evens.count(&rt);
    let odds_count = odds.count(&rt);
    let total_count = kc.collection().count(&rt);

    let expected_evens = (0..KEY_SPACE).filter(|k| k % 2 == 0).count() as u64;
    let expected_odds = KEY_SPACE - expected_evens;

    let stop = Arc::new(AtomicBool::new(false));
    let mut readers = Vec::new();
    for _ in 0..4 {
        let rt = Arc::clone(&rt);
        let groups_version = groups.version_node();
        let stop = Arc::clone(&stop);
        readers.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let _ = rt.get(groups_version);
                assert_eq!(
                    rt.get(total_count),
                    KEY_SPACE,
                    "total count dipped during a replace"
                );
                assert_eq!(
                    rt.get(evens_count),
                    expected_evens,
                    "even-group count dipped during a replace"
                );
                assert_eq!(
                    rt.get(odds_count),
                    expected_odds,
                    "odd-group count dipped during a replace"
                );
            }
        }));
    }

    let rounds = iters(2000);
    let mut version = vec![1_u64; KEY_SPACE as usize];
    for i in 0..rounds {
        let key = (i as u64) % KEY_SPACE;
        version[key as usize] += 1;
        kc.upsert(&rt, key, version[key as usize], key);
    }

    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }

    assert_eq!(kc.len() as u64, KEY_SPACE);
    let _ = rt.get(groups.version_node());
    assert_eq!(rt.get(total_count), KEY_SPACE);
}

#[test]
fn mixed_upsert_remove() {
    let rt: Arc<Runtime> = Arc::new(Runtime::new());
    let kc: Arc<KeyedCollection<u64, u64>> = Arc::new(KeyedCollection::new(&rt));
    let count = kc.collection().count(&rt);

    let stop = Arc::new(AtomicBool::new(false));
    let mut readers = Vec::new();
    for _ in 0..4 {
        let rt = Arc::clone(&rt);
        let stop = Arc::clone(&stop);
        readers.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let n = rt.get(count);
                assert!(
                    n <= KEY_SPACE,
                    "count {n} exceeded key-space size {KEY_SPACE}"
                );
            }
        }));
    }

    // Single writer, so a plain local map is the ground truth: no lock
    // needed, nothing else touches it.
    let mut live: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
    let rounds = iters(4000);
    let mut seed = 0x9e3779b97f4a7c15_u64;
    for i in 0..rounds {
        // xorshift64*, deterministic and dependency-free.
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let key = seed % KEY_SPACE;
        let version = i as u64 + 1;
        if seed.is_multiple_of(3) && live.contains_key(&key) {
            kc.remove(&rt, &key);
            live.remove(&key);
        } else {
            kc.upsert(&rt, key, version, key);
            live.insert(key, version);
        }
    }

    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }

    assert_eq!(kc.len(), live.len());
    for key in live.keys() {
        assert!(kc.get(key).is_some());
    }
}
