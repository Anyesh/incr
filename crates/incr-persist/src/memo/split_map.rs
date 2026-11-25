//! A hash map that grows one bucket at a time (linear hashing), so that no
//! insert rehashes more than about one bucket's entries, whatever the
//! map's size. A memo flight inserts into these maps between looks at the
//! clock, where a whole-map rehash would hold the caller in one piece.

use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

/// Entries per bucket, on average, before the next bucket splits.
const SPLIT_AT: usize = 1024;

/// An unseeded hash: the keys are runtime handles, which nothing outside
/// the process chooses, so it needs no defence against chosen collisions
/// and costs a few multiplies instead of SipHash.
#[derive(Default)]
struct Mix(u64);

impl Hasher for Mix {
    fn finish(&self) -> u64 {
        let mut hash = self.0;
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        hash ^ (hash >> 33)
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    fn write_u32(&mut self, word: u32) {
        self.write_u64(u64::from(word));
    }

    fn write_u64(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(29) ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

/// [`Mix`] rotated by half, for a bucket's own map: the map probes by the
/// low bits and tags by the top seven, while a bucket's keys share the low
/// bits that address the bucket.
#[derive(Default)]
struct Rotated(Mix);

impl Hasher for Rotated {
    fn finish(&self) -> u64 {
        self.0.finish().rotate_right(32)
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }

    fn write_u32(&mut self, word: u32) {
        self.0.write_u32(word);
    }

    fn write_u64(&mut self, word: u64) {
        self.0.write_u64(word);
    }
}

type Bucket<K, V> = HashMap<K, V, BuildHasherDefault<Rotated>>;

fn address<K: Hash>(key: &K) -> usize {
    BuildHasherDefault::<Mix>::default().hash_one(key) as usize
}

pub(crate) struct SplitMap<K, V> {
    buckets: Vec<Bucket<K, V>>,
    /// The bucket count when this round of splits began, a power of two.
    round: usize,
    /// The next bucket to split, below `round`.
    next: usize,
    len: usize,
}

impl<K, V> Default for SplitMap<K, V> {
    fn default() -> Self {
        SplitMap {
            buckets: vec![Bucket::default()],
            round: 1,
            next: 0,
            len: 0,
        }
    }
}

impl<K: Hash + Eq, V> SplitMap<K, V> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    fn bucket(&self, key: &K) -> usize {
        let hash = address(key);
        let at = hash & (self.round - 1);
        if at < self.next {
            hash & (2 * self.round - 1)
        } else {
            at
        }
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.buckets[self.bucket(key)].get(key)
    }

    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        let at = self.bucket(&key);
        let old = self.buckets[at].insert(key, value);
        if old.is_none() {
            self.len += 1;
            if self.len > SPLIT_AT * self.buckets.len() {
                self.split();
            }
        }
        old
    }

    /// Moves the keys of bucket `next` that address the new bucket under
    /// the next round's mask into it.
    fn split(&mut self) {
        let from = self.next;
        let mask = 2 * self.round - 1;
        let moved: Bucket<K, V> = self.buckets[from]
            .extract_if(|key, _| address(key) & mask != from)
            .collect();
        self.buckets.push(moved);
        self.next += 1;
        if self.next == self.round {
            self.round *= 2;
            self.next = 0;
        }
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&K, &mut V) -> bool) {
        for at in 0..self.buckets.len() {
            self.retain_bucket(at, &mut keep);
        }
    }

    pub(crate) fn buckets(&self) -> usize {
        self.buckets.len()
    }

    /// [`Self::retain`] for bucket `at` alone; returns how many entries it
    /// looked at. A split moves keys only to a new bucket after every
    /// existing one, so retaining buckets `0..buckets()` in order, across
    /// inserts, looks at every key present when it began.
    pub(crate) fn retain_bucket(
        &mut self,
        at: usize,
        mut keep: impl FnMut(&K, &mut V) -> bool,
    ) -> usize {
        let bucket = &mut self.buckets[at];
        let before = bucket.len();
        bucket.retain(|key, value| keep(key, value));
        self.len -= before - bucket.len();
        before
    }
}

#[cfg(test)]
#[path = "split_map_tests.rs"]
mod tests;
