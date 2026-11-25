use super::*;

#[test]
fn every_key_stays_reachable_across_splits() {
    let mut map = SplitMap::new();
    for k in 0..50_000_u32 {
        assert_eq!(map.insert(k, k * 2), None);
    }
    assert_eq!(map.insert(7, 0), Some(14));
    assert_eq!(map.len(), 50_000);
    assert!(map.buckets() > 32);
    for k in 1..50_000_u32 {
        if k != 7 {
            assert_eq!(map.get(&k), Some(&(k * 2)));
        }
    }
    assert_eq!(map.get(&50_000), None);
}

#[test]
fn no_bucket_holds_more_than_a_few_splits_worth() {
    let mut map = SplitMap::new();
    for k in 0..200_000_u64 {
        map.insert(k, ());
        if k % 1000 == 0 {
            let most = map.buckets.iter().map(HashMap::len).max().unwrap();
            assert!(most <= 4 * SPLIT_AT, "a bucket holds {most} keys");
        }
    }
}

#[test]
fn retaining_bucket_by_bucket_sees_every_key_despite_inserts() {
    let mut map = SplitMap::new();
    for k in 0..20_000_u32 {
        map.insert(k, ());
    }
    let mut seen = std::collections::HashSet::new();
    let mut at = 0;
    let mut next = 20_000_u32;
    while at < map.buckets() {
        map.retain_bucket(at, |&k, _| {
            seen.insert(k);
            k % 3 != 0
        });
        at += 1;
        for _ in 0..700 {
            map.insert(next, ());
            next += 1;
        }
    }
    assert!((0..20_000).all(|k| seen.contains(&k)));
    assert!((0..20_000).all(|k| (map.get(&k).is_some()) == (k % 3 != 0)));
    let present = (0..next).filter(|k| map.get(k).is_some()).count();
    assert_eq!(map.len(), present);
}
