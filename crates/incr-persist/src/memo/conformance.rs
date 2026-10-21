//! The contract every [`MemoBackend`] must meet, as runnable checks.
//!
//! Implement [`MemoHarness`] for a backend and invoke
//! [`memo_backend_conformance!`](crate::memo_backend_conformance) in a test
//! file to get one `#[test]` per case. Each case is also a public function
//! taking the harness by value. Cases panic on a contract violation. Crash
//! durability (a kill between the temp-file write and the rename, a torn
//! frame) is backend-specific and tested by each backend itself.

use std::sync::Barrier;
use std::thread;

use super::{EncodedEntry, Entry, GenHeader, GenId, Generation, MemoBackend, StoredDep, TableDesc};
use crate::Error;

/// Creates memo stores and opens handles onto them for the suite.
pub trait MemoHarness {
    type Backend: MemoBackend + 'static;
    /// Identifies one store, such as a directory or a database path.
    type Store;

    fn new_store(&mut self) -> Self::Store;

    /// Opens a handle on `store`. Several handles on one store may be open
    /// at once, as several processes would hold them.
    fn open(&mut self, store: &Self::Store) -> Self::Backend;

    /// Whether [`MemoBackend::lock_checkpoint`] excludes other handles
    /// with [`Error::Busy`]. Lock cases are skipped when this is `false`.
    fn exclusive_lock(&self) -> bool;
}

/// Expands to one `#[test]` per memo backend conformance case. Requires the
/// `backend-tests` feature.
///
/// ```ignore
/// incr_persist::memo_backend_conformance!(MyHarness::new());
/// ```
#[macro_export]
macro_rules! memo_backend_conformance {
    ($harness:expr) => {
        $crate::memo_backend_conformance!(@cases $harness;
            fresh_store_loads_nothing,
            written_generation_loads_and_reads_back,
            empty_generation_roundtrips,
            generation_survives_reopen,
            write_from_a_stale_base_is_busy,
            one_of_two_racing_writers_publishes,
            older_generation_stays_readable_until_removed,
            checkpoint_lock_excludes_other_handles,
            checkpoint_lock_removes_generations_below_current,
        );
    };
    (@cases $harness:expr; $($case:ident),+ $(,)?) => {
        $(
            #[test]
            fn $case() {
                $crate::memo::conformance::$case($harness);
            }
        )+
    };
}

fn header(seq: u64) -> GenHeader {
    GenHeader {
        build: 0xb111d,
        seq,
        tables: vec![
            TableDesc {
                name: "fib".into(),
                version: 1,
                key_type: "u64".into(),
                value_type: "u64".into(),
            },
            TableDesc {
                name: "parse".into(),
                version: 2,
                key_type: "alloc::string::String".into(),
                value_type: "alloc::vec::Vec<u8>".into(),
            },
        ],
    }
}

fn entries(count: u8) -> Vec<EncodedEntry> {
    (0..count)
        .map(|i| {
            let entry = Entry {
                table: u32::from(i % 2),
                key: vec![i; usize::from(i) + 1],
                value: vec![i.wrapping_mul(31); usize::from(i) * 7],
                deps: vec![StoredDep::Input {
                    key: format!("in{i}"),
                    fp: u128::from(i),
                }],
                last_used: u64::from(i),
            };
            EncodedEntry::new(&entry).expect("encode")
        })
        .collect()
}

fn write<B: MemoBackend>(
    backend: &B,
    base: Option<GenId>,
    id: u64,
    written: &[EncodedEntry],
) -> Generation {
    backend
        .write_generation(base, GenId(id), &header(id), written)
        .expect("write_generation failed")
}

fn assert_holds<B: MemoBackend>(backend: &B, generation: &Generation, written: &[EncodedEntry]) {
    assert_eq!(generation.items().len(), written.len());
    for entry in written {
        let found = generation.find(entry.table, entry.key_hash);
        assert_eq!(found.len(), 1, "one item per written entry");
        let item = found[0];
        assert_eq!(item.len as usize, entry.frame.len());
        let frame = backend
            .read_entry(generation.id(), &item)
            .expect("read_entry failed")
            .expect("a current generation must be readable");
        assert_eq!(frame, entry.frame, "frames come back byte for byte");
    }
}

/// A store no generation was written to loads `None`.
pub fn fresh_store_loads_nothing<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    assert!(h.open(&store).load().expect("load failed").is_none());
}

/// `write_generation` returns what `load` then returns, and every frame
/// reads back unchanged through the returned items.
pub fn written_generation_loads_and_reads_back<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    let written = entries(6);
    let generation = write(&backend, None, 1, &written);
    assert_eq!(generation.id(), GenId(1));
    assert_eq!(generation.header(), &header(1));
    assert_holds(&backend, &generation, &written);

    let loaded = backend.load().expect("load failed").expect("a generation");
    assert_eq!(loaded.id(), generation.id());
    assert_eq!(loaded.header(), generation.header());
    assert_eq!(loaded.items(), generation.items());
    assert_holds(&backend, &loaded, &written);
}

/// A generation with no entries is still a generation, with its header.
pub fn empty_generation_roundtrips<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    write(&backend, None, 1, &[]);
    let loaded = backend.load().expect("load failed").expect("a generation");
    assert_eq!(loaded.header(), &header(1));
    assert!(loaded.items().is_empty());
}

/// A written generation is durable: a new handle loads and reads it.
pub fn generation_survives_reopen<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let written = entries(4);
    let first = h.open(&store);
    write(&first, None, 1, &written);
    drop(first);
    let backend = h.open(&store);
    let loaded = backend.load().expect("load failed").expect("a generation");
    assert_eq!(loaded.id(), GenId(1));
    assert_eq!(loaded.header(), &header(1));
    assert_holds(&backend, &loaded, &written);
}

/// A write whose base is no longer current fails with `Busy` and leaves
/// the current generation as it was.
pub fn write_from_a_stale_base_is_busy<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    write(&backend, None, 1, &entries(2));
    let stale = backend.write_generation(None, GenId(1), &header(1), &entries(3));
    assert!(matches!(stale, Err(Error::Busy)), "got {stale:?}");
    let stale = backend.write_generation(Some(GenId(7)), GenId(8), &header(8), &entries(3));
    assert!(matches!(stale, Err(Error::Busy)), "got {stale:?}");
    let loaded = backend.load().expect("load failed").expect("a generation");
    assert_eq!(loaded.id(), GenId(1));
    assert_eq!(loaded.items().len(), 2);
    write(&backend, Some(GenId(1)), 2, &entries(3));
    assert_eq!(backend.load().unwrap().unwrap().id(), GenId(2));
}

/// Two handles that write from the same base at once: exactly one
/// publishes, the other gets `Busy`, and `load` returns the winner's.
pub fn one_of_two_racing_writers_publishes<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let a = h.open(&store);
    let b = h.open(&store);
    write(&a, None, 1, &entries(1));
    let barrier = Barrier::new(2);
    let results: Vec<_> = thread::scope(|scope| {
        let race = |backend: &H::Backend, count: u8| {
            barrier.wait();
            backend
                .write_generation(Some(GenId(1)), GenId(2), &header(2), &entries(count))
                .map(|generation| (count, generation.items().len()))
        };
        let (first, second) = (&a, &b);
        let a = scope.spawn(move || race(first, 2));
        let b = scope.spawn(move || race(second, 3));
        vec![a.join().unwrap(), b.join().unwrap()]
    });
    let winners: Vec<_> = results.iter().filter_map(|r| r.as_ref().ok()).collect();
    assert_eq!(winners.len(), 1, "results: {results:?}");
    assert!(results.iter().any(|r| matches!(r, Err(Error::Busy))));
    let loaded = a.load().unwrap().unwrap();
    assert_eq!(loaded.id(), GenId(2));
    assert_eq!(loaded.items().len(), winners[0].1);
}

/// Replacing a generation leaves the old one readable until it is removed;
/// after that its reads return `None`. Removing it again is not an error.
pub fn older_generation_stays_readable_until_removed<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    let old_entries = entries(3);
    let old = write(&backend, None, 1, &old_entries);
    let new = write(&backend, Some(GenId(1)), 2, &entries(5));
    assert_eq!(backend.load().unwrap().unwrap().id(), new.id());
    assert_holds(&backend, &old, &old_entries);

    backend.remove_generation(old.id()).expect("remove failed");
    for item in old.items() {
        assert_eq!(
            backend
                .read_entry(old.id(), item)
                .expect("read_entry failed"),
            None
        );
    }
    backend
        .remove_generation(old.id())
        .expect("a second remove failed");
    assert_eq!(backend.load().unwrap().unwrap().id(), new.id());
    let other = h.open(&store);
    assert_eq!(other.load().unwrap().unwrap().id(), new.id());
}

/// While one handle holds the checkpoint lock another handle's
/// `lock_checkpoint` returns `Busy`; once it is dropped the lock is free.
pub fn checkpoint_lock_excludes_other_handles<H: MemoHarness>(mut h: H) {
    if !h.exclusive_lock() {
        return;
    }
    let store = h.new_store();
    let a = h.open(&store);
    let b = h.open(&store);
    let guard = a.lock_checkpoint().expect("lock failed");
    assert!(matches!(b.lock_checkpoint(), Err(Error::Busy)));
    drop(guard);
    let guard = b.lock_checkpoint().expect("lock after release failed");
    assert!(matches!(a.lock_checkpoint(), Err(Error::Busy)));
    drop(guard);
}

/// Taking the checkpoint lock removes every generation below the current
/// one, such as one whose writer died before removing it.
pub fn checkpoint_lock_removes_generations_below_current<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    let old = write(&backend, None, 1, &entries(3));
    let written = entries(4);
    write(&backend, Some(GenId(1)), 2, &written);
    drop(backend.lock_checkpoint().expect("lock failed"));

    let other = h.open(&store);
    for item in old.items() {
        assert_eq!(
            other.read_entry(old.id(), item).expect("read_entry failed"),
            None,
            "generation 1 outlived the checkpoint lock"
        );
    }
    let loaded = other.load().unwrap().unwrap();
    assert_eq!(loaded.id(), GenId(2));
    assert_holds(&other, &loaded, &written);
}
