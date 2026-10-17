//! Open-addressing multiset keyed by caller-supplied hashes.
//!
//! The table stores each element's hash next to it, so `reserve` and the
//! rehash it may trigger move entries by their stored hashes and never run
//! user code. Lookup is the only operation that calls user `Eq`; every
//! mutation takes an index that a lookup returned (for an absent value,
//! the first free slot on its probe run), so a caller can run all user
//! code first and then mutate without any user code running in between.
//! `T` carries no `Hash` bound at all, which makes "the table never
//! hashes" a compile-time fact.
//!
//! Removal leaves a tombstone (or an empty slot when no probe run can pass
//! through it) and never moves another entry, and only `reserve` rehashes,
//! so indices stay valid until the next `reserve`.

use std::mem::{self, MaybeUninit};

const EMPTY: u8 = 0;
const TOMBSTONE: u8 = 1;
const FULL: u8 = 0x80;

/// Full slots store `FULL | top 7 hash bits`, so most probe mismatches
/// are rejected from the control byte without touching the entry.
fn tag(hash: u64) -> u8 {
    FULL | (hash >> 57) as u8
}

fn load_limit(capacity: usize) -> usize {
    capacity / 4 * 3
}

struct Entry<T> {
    hash: u64,
    count: usize,
    value: T,
}

pub(crate) struct HashedMultiset<T> {
    // Invariant: `ctrl[i] & FULL != 0` exactly when `entries[i]` is
    // initialised. Both slices have the same power-of-two length (or are
    // empty), and at least one slot is EMPTY whenever they are not empty,
    // so every probe loop terminates.
    ctrl: Box<[u8]>,
    entries: Box<[MaybeUninit<Entry<T>>]>,
    len: usize,
    growth_left: usize,
}

/// Where one change lands, as found by [`HashedMultiset::locate`]. Valid
/// for the multiset that produced it until its next `reserve`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Located {
    remove: Option<usize>,
    add: Option<AddAt>,
}

/// Where one added occurrence lands, as found by
/// [`HashedMultiset::locate_add`]. Valid until the multiset next changes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum AddAt {
    Existing(usize),
    /// No entry equals the value; `slot` is the first free slot on its
    /// probe run, where it is stored.
    Absent {
        hash: u64,
        slot: usize,
    },
}

impl Located {
    pub(crate) fn removes(&self) -> bool {
        self.remove.is_some()
    }
}

impl<T> HashedMultiset<T> {
    pub(crate) fn new() -> Self {
        Self {
            ctrl: Box::new([]),
            entries: Box::new([]),
            len: 0,
            growth_left: 0,
        }
    }

    /// Number of distinct elements.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Distinct elements with their multiplicities, in table order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&T, usize)> {
        self.ctrl
            .iter()
            .zip(self.entries.iter())
            .filter(|(c, _)| **c & FULL != 0)
            // SAFETY: the slot is full, so its entry is initialised.
            .map(|(_, e)| unsafe { e.assume_init_ref() })
            .map(|e| (&e.value, e.count))
    }

    /// Ensure `additional` more distinct elements can be inserted without
    /// a rehash. Runs no user code.
    pub(crate) fn reserve(&mut self, additional: usize) {
        if additional > self.growth_left {
            self.rehash(additional);
        }
    }

    #[cold]
    fn rehash(&mut self, additional: usize) {
        let need = self
            .len
            .checked_add(additional)
            .expect("incr-core: collection capacity overflow");
        let limit = load_limit(self.ctrl.len());
        // Purging tombstones at the same size is enough when it frees at
        // least half the limit, which keeps rehashes amortised O(1).
        let capacity = if need <= limit / 2 {
            self.ctrl.len()
        } else {
            let need = need.max(limit + 1);
            let mut capacity = 8_usize;
            while load_limit(capacity) < need {
                capacity = capacity
                    .checked_mul(2)
                    .expect("incr-core: collection capacity overflow");
            }
            capacity
        };
        let mask = capacity - 1;
        let mut ctrl = vec![EMPTY; capacity].into_boxed_slice();
        let mut entries = Box::<[Entry<T>]>::new_uninit_slice(capacity);
        for i in 0..self.ctrl.len() {
            if self.ctrl[i] & FULL == 0 {
                continue;
            }
            // SAFETY: the slot is full. Marking it empty right away hands
            // ownership to the new table, so the old one never drops it.
            let entry = unsafe { self.entries[i].assume_init_read() };
            let c = mem::replace(&mut self.ctrl[i], EMPTY);
            let mut j = entry.hash as usize & mask;
            while ctrl[j] != EMPTY {
                j = (j + 1) & mask;
            }
            ctrl[j] = c;
            entries[j].write(entry);
        }
        self.ctrl = ctrl;
        self.entries = entries;
        self.growth_left = load_limit(capacity) - self.len;
    }

    /// Index of the entry equal to `value`, which must hash to `hash`.
    /// The only operation that runs user code (`T::eq`), and it changes
    /// nothing.
    pub(crate) fn find(&self, hash: u64, value: &T) -> Option<usize>
    where
        T: PartialEq,
    {
        if self.ctrl.is_empty() {
            return None;
        }
        self.probe(hash, value).ok()
    }

    /// `Ok` with the index of the entry equal to `value`, or `Err` with
    /// the first free slot on its probe run, the slot an insert of it
    /// takes. Requires a non-empty table.
    #[inline(always)]
    fn probe(&self, hash: u64, value: &T) -> Result<usize, usize>
    where
        T: PartialEq,
    {
        let mask = self.ctrl.len() - 1;
        let tag = tag(hash);
        let mut i = hash as usize & mask;
        let mut free = usize::MAX;
        loop {
            let c = self.ctrl[i];
            if c == tag {
                // SAFETY: a tag byte has FULL set, so the entry is initialised.
                let e = unsafe { self.entries[i].assume_init_ref() };
                if e.hash == hash && e.value == *value {
                    return Ok(i);
                }
            } else if c & FULL == 0 {
                if c == EMPTY {
                    return Err(if free == usize::MAX { i } else { free });
                }
                if free == usize::MAX {
                    free = i;
                }
            }
            i = (i + 1) & mask;
        }
    }

    /// Requires a prior `reserve(1)`, so the table is not empty.
    fn locate_one(&self, hash: u64, value: &T) -> AddAt
    where
        T: PartialEq,
    {
        match self.probe(hash, value) {
            Ok(i) => AddAt::Existing(i),
            Err(slot) => AddAt::Absent { hash, slot },
        }
    }

    fn entry_mut(&mut self, i: usize) -> &mut Entry<T> {
        assert!(
            self.ctrl[i] & FULL != 0,
            "incr-core: multiset index {i} is not a live entry"
        );
        // SAFETY: checked full just above.
        unsafe { self.entries[i].assume_init_mut() }
    }

    fn count(&mut self, i: usize) -> usize {
        self.entry_mut(i).count
    }

    fn increment(&mut self, i: usize) {
        self.entry_mut(i).count += 1;
    }

    fn swap_value(&mut self, i: usize, new: T) -> T {
        mem::replace(&mut self.entry_mut(i).value, new)
    }

    /// Decrement the count at `i`, removing the entry and handing its
    /// value back (for the caller to drop) when the count reaches zero.
    /// Runs no user code.
    pub(crate) fn decrement_or_remove(&mut self, i: usize) -> Option<T> {
        let entry = self.entry_mut(i);
        if entry.count > 1 {
            entry.count -= 1;
            return None;
        }
        let next = (i + 1) & (self.ctrl.len() - 1);
        // An entry's probe run covers every slot from its home to its own
        // position, so a run can only pass through `i` if it continues to
        // `i + 1`. With `i + 1` empty, `i` can be emptied too.
        self.ctrl[i] = if self.ctrl[next] == EMPTY {
            self.growth_left += 1;
            EMPTY
        } else {
            TOMBSTONE
        };
        self.len -= 1;
        // SAFETY: the slot was full and is now marked free, so this read
        // takes the only ownership of the entry.
        Some(unsafe { self.entries[i].assume_init_read() }.value)
    }

    /// Insert `value`, which no entry equals, with count 1 at `i`, the
    /// free slot its lookup returned. Requires a prior `reserve(1)`.
    fn insert_absent(&mut self, hash: u64, i: usize, value: T) {
        assert!(
            self.growth_left > 0,
            "incr-core: multiset insert without a reserve"
        );
        assert!(
            self.ctrl[i] & FULL == 0,
            "incr-core: multiset slot {i} is not free"
        );
        if self.ctrl[i] == EMPTY {
            self.growth_left -= 1;
        }
        self.entries[i].write(Entry {
            hash,
            count: 1,
            value,
        });
        self.ctrl[i] = tag(hash);
        self.len += 1;
    }

    /// Locate one change: removing one occurrence of `remove` and adding
    /// one occurrence of `add`, each given with its hash. Reserves room
    /// for the add, then runs the lookups (user `Eq`). Changes no
    /// element, so a panic here leaves the multiset as it was.
    pub(crate) fn locate(&mut self, remove: Option<(u64, &T)>, add: Option<(u64, &T)>) -> Located
    where
        T: PartialEq,
    {
        if add.is_some() {
            self.reserve(1);
        }
        let remove = remove.and_then(|(hash, value)| self.find(hash, value));
        let add = add.map(|(hash, value)| self.locate_one(hash, value));
        Located { remove, add }
    }

    /// Locate one added occurrence of `value`: reserves room, then runs
    /// the lookup (user `Eq`). Changes no element.
    #[inline]
    pub(crate) fn locate_add(&mut self, hash: u64, value: &T) -> AddAt
    where
        T: PartialEq,
    {
        self.reserve(1);
        self.locate_one(hash, value)
    }

    /// Add one occurrence at `at`. Runs no user code. Hands `value` back
    /// when an equal element was already stored, for the caller to drop
    /// when it is safe to run user `Drop`.
    #[inline]
    pub(crate) fn add(&mut self, at: AddAt, value: T) -> Option<T> {
        match at {
            AddAt::Existing(i) => {
                self.increment(i);
                Some(value)
            }
            AddAt::Absent { hash, slot } => {
                self.insert_absent(hash, slot, value);
                None
            }
        }
    }

    /// Apply a located change, with `add` the value to store if the added
    /// element is new. Runs no user code. Returns the values the change
    /// displaced, for the caller to drop when it is safe to run user
    /// `Drop`: the unused `add` when its element was already present, and
    /// the stored value when the last occurrence is removed.
    ///
    /// When remove and add name the same entry at count 1, the stored
    /// value is swapped for `add`, the same result as a removal followed
    /// by an insert; at a higher count the stored value stays.
    pub(crate) fn apply(&mut self, at: Located, add: Option<T>) -> [Option<T>; 2] {
        assert_eq!(
            at.add.is_some(),
            add.is_some(),
            "incr-core: multiset change applied with a mismatched value"
        );
        let mut displaced = [None, None];
        match (at.add, add) {
            (Some(AddAt::Existing(i)), Some(value)) if at.remove == Some(i) => {
                displaced[0] = Some(if self.count(i) == 1 {
                    self.swap_value(i, value)
                } else {
                    value
                });
                return displaced;
            }
            (Some(at), Some(value)) => displaced[0] = self.add(at, value),
            _ => {}
        }
        if let Some(i) = at.remove {
            displaced[1] = self.decrement_or_remove(i);
        }
        displaced
    }
}

impl<T> Default for HashedMultiset<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for HashedMultiset<T> {
    fn drop(&mut self) {
        if !mem::needs_drop::<T>() {
            return;
        }
        for (c, e) in self.ctrl.iter().zip(self.entries.iter_mut()) {
            if *c & FULL != 0 {
                // SAFETY: the slot is full, and the table is never used again.
                unsafe { e.assume_init_drop() };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    /// Equal by `key` only, so which copy the table keeps is observable
    /// through `id`. The boxed id lets Miri catch a double drop or a leak.
    #[derive(Debug)]
    struct V {
        key: u8,
        id: Box<u32>,
    }

    impl PartialEq for V {
        fn eq(&self, other: &Self) -> bool {
            self.key == other.key
        }
    }

    fn v(key: u8, id: u32) -> V {
        V {
            key,
            id: Box::new(id),
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Hashing {
        // Few homes and identical tags: long merged probe runs, and lookups
        // that reach `eq` on unequal values.
        Clustered,
        Spread,
    }

    fn hash(hashing: Hashing, key: u8) -> u64 {
        match hashing {
            Hashing::Clustered => u64::from(key % 3),
            Hashing::Spread => u64::from(key).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[derive(Clone, Debug)]
    enum Op {
        Insert(u8, u32),
        Remove(u8),
        Replace(Option<u8>, u8, u32),
        Reserve(usize),
    }

    fn op() -> impl Strategy<Value = Op> {
        let key = 0_u8..24;
        prop_oneof![
            3 => (key.clone(), any::<u32>()).prop_map(|(k, id)| Op::Insert(k, id)),
            2 => key.clone().prop_map(Op::Remove),
            3 => (proptest::option::of(key.clone()), key, any::<u32>())
                .prop_map(|(old, k, id)| Op::Replace(old, k, id)),
            1 => (0_usize..40).prop_map(Op::Reserve),
        ]
    }

    /// Model: key -> (count, id of the stored copy).
    type Model = HashMap<u8, (usize, u32)>;

    fn change(
        set: &mut HashedMultiset<V>,
        hashing: Hashing,
        remove: Option<u8>,
        add: Option<(u8, u32)>,
    ) -> (bool, Vec<u32>) {
        let remove = remove.map(|k| v(k, u32::MAX));
        let add = add.map(|(k, id)| v(k, id));
        let at = set.locate(
            remove.as_ref().map(|r| (hash(hashing, r.key), r)),
            add.as_ref().map(|a| (hash(hashing, a.key), a)),
        );
        let removed = at.removes();
        let mut displaced: Vec<u32> = set
            .apply(at, add)
            .into_iter()
            .flatten()
            .map(|d| *d.id)
            .collect();
        displaced.sort_unstable();
        (removed, displaced)
    }

    /// The single-sided paths the collection mutations take.
    fn add_one(set: &mut HashedMultiset<V>, hashing: Hashing, k: u8, id: u32) -> (bool, Vec<u32>) {
        let value = v(k, id);
        let at = set.locate_add(hash(hashing, k), &value);
        let displaced = set.add(at, value).map(|d| *d.id);
        (false, displaced.into_iter().collect())
    }

    fn remove_one(set: &mut HashedMultiset<V>, hashing: Hashing, k: u8) -> (bool, Vec<u32>) {
        let Some(i) = set.find(hash(hashing, k), &v(k, u32::MAX)) else {
            return (false, Vec::new());
        };
        let displaced = set.decrement_or_remove(i).map(|d| *d.id);
        (true, displaced.into_iter().collect())
    }

    fn model_change(
        model: &mut Model,
        remove: Option<u8>,
        add: Option<(u8, u32)>,
    ) -> (bool, Vec<u32>) {
        let removed = remove.is_some_and(|k| model.contains_key(&k));
        let mut displaced = Vec::new();
        if let (Some(old), Some((key, id))) = (remove.filter(|_| removed), add) {
            if old == key {
                let entry = model.get_mut(&key).unwrap();
                if entry.0 == 1 {
                    displaced.push(entry.1);
                    entry.1 = id;
                } else {
                    displaced.push(id);
                }
                return (true, displaced);
            }
        }
        if let Some((key, id)) = add {
            match model.get_mut(&key) {
                Some(entry) => {
                    entry.0 += 1;
                    displaced.push(id);
                }
                None => {
                    model.insert(key, (1, id));
                }
            }
        }
        if let Some(key) = remove.filter(|_| removed) {
            let entry = model.get_mut(&key).unwrap();
            entry.0 -= 1;
            if entry.0 == 0 {
                displaced.push(entry.1);
                model.remove(&key);
            }
        }
        displaced.sort_unstable();
        (removed, displaced)
    }

    fn assert_matches(set: &HashedMultiset<V>, hashing: Hashing, model: &Model) {
        assert_eq!(set.len(), model.len());
        let mut seen: Vec<(u8, usize, u32)> = set.iter().map(|(v, c)| (v.key, c, *v.id)).collect();
        seen.sort_unstable();
        let mut expected: Vec<(u8, usize, u32)> =
            model.iter().map(|(&k, &(c, id))| (k, c, id)).collect();
        expected.sort_unstable();
        assert_eq!(seen, expected);
        for key in 0..24 {
            let found = set.find(hash(hashing, key), &v(key, 0));
            assert_eq!(found.is_some(), model.contains_key(&key), "find({key})");
        }
        let full = set.ctrl.iter().filter(|c| **c & FULL != 0).count();
        let tombstones = set.ctrl.iter().filter(|c| **c == TOMBSTONE).count();
        assert_eq!(full, set.len);
        assert_eq!(
            full + tombstones + set.growth_left,
            load_limit(set.ctrl.len())
        );
    }

    fn run(hashing: Hashing, ops: &[Op]) {
        let mut set = HashedMultiset::new();
        let mut model = Model::new();
        for op in ops {
            match *op {
                Op::Insert(k, id) => {
                    let got = add_one(&mut set, hashing, k, id);
                    assert_eq!(got, model_change(&mut model, None, Some((k, id))), "{op:?}");
                }
                Op::Remove(k) => {
                    let got = remove_one(&mut set, hashing, k);
                    assert_eq!(got, model_change(&mut model, Some(k), None), "{op:?}");
                }
                Op::Replace(old, k, id) => {
                    let got = change(&mut set, hashing, old, Some((k, id)));
                    assert_eq!(got, model_change(&mut model, old, Some((k, id))), "{op:?}");
                }
                Op::Reserve(n) => {
                    let before = set.ctrl.len();
                    set.reserve(n);
                    assert!(set.growth_left >= n);
                    assert!(set.ctrl.len() >= before, "reserve shrank the table");
                }
            }
            assert_matches(&set, hashing, &model);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: if cfg!(miri) { 6 } else { 512 },
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        #[test]
        fn matches_a_hashmap_model(
            clustered in any::<bool>(),
            ops in proptest::collection::vec(op(), 0..if cfg!(miri) { 40 } else { 200 }),
        ) {
            let hashing = if clustered { Hashing::Clustered } else { Hashing::Spread };
            run(hashing, &ops);
        }
    }

    #[test]
    fn replace_with_an_equal_value_at_count_one_swaps_the_stored_copy() {
        for hashing in [Hashing::Clustered, Hashing::Spread] {
            run(
                hashing,
                &[
                    Op::Insert(5, 1),
                    Op::Replace(Some(5), 5, 2),
                    Op::Replace(Some(5), 5, 3),
                ],
            );
        }
    }

    #[test]
    fn replace_with_an_equal_value_at_count_two_keeps_the_stored_copy() {
        for hashing in [Hashing::Clustered, Hashing::Spread] {
            run(
                hashing,
                &[
                    Op::Insert(5, 1),
                    Op::Insert(5, 2),
                    Op::Replace(Some(5), 5, 3),
                    Op::Remove(5),
                ],
            );
        }
    }

    #[test]
    fn replace_where_old_sits_earlier_in_new_probe_run() {
        // Keys 0, 3 and 6 share a home under clustered hashing. 0 takes the
        // home slot and 3 the next, so replacing 0 by 6 puts 6 after both,
        // then tombstones the home slot that 6 and 3 are found through.
        let ops = [
            Op::Insert(0, 1),
            Op::Insert(3, 2),
            Op::Replace(Some(0), 6, 3),
            Op::Replace(Some(3), 0, 4),
            Op::Replace(Some(6), 3, 5),
            Op::Insert(9, 6),
        ];
        run(Hashing::Clustered, &ops);
    }

    #[test]
    fn a_located_index_stays_valid_across_the_add_it_was_located_with() {
        let mut set = HashedMultiset::new();
        let hashing = Hashing::Clustered;
        for (k, id) in [(0, 0), (3, 1)] {
            change(&mut set, hashing, None, Some((k, id)));
        }
        let home = hash(hashing, 0) as usize & (set.ctrl.len() - 1);
        assert_eq!(set.find(hash(hashing, 0), &v(0, 0)), Some(home));
        let (removed, displaced) = change(&mut set, hashing, Some(0), Some((6, 2)));
        assert!(removed);
        assert_eq!(displaced, vec![0]);
        assert_eq!(set.ctrl[home], TOMBSTONE);
        assert!(set.find(hash(hashing, 6), &v(6, 0)).is_some());
        assert!(set.find(hash(hashing, 3), &v(3, 0)).is_some());
    }

    #[test]
    fn growth_rehash_keeps_every_entry() {
        let mut model = Model::new();
        let mut set = HashedMultiset::new();
        for k in 0..24 {
            let got = change(&mut set, Hashing::Clustered, None, Some((k, u32::from(k))));
            assert_eq!(got, model_change(&mut model, None, Some((k, u32::from(k)))));
        }
        assert_matches(&set, Hashing::Clustered, &model);
        assert!(set.ctrl.len() >= 32);
    }
}
