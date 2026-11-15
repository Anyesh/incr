//! The contract every [`MemoBackend`] must meet, as runnable checks.
//!
//! Implement [`MemoHarness`] for a backend and invoke
//! [`memo_backend_conformance!`](crate::memo_backend_conformance) in a test
//! file to get one `#[test]` per case. Each case is also a public function
//! taking the harness by value. Cases panic on a contract violation. Crash
//! durability (a kill between the temp-file write and the rename, a torn
//! frame) is backend-specific and tested by each backend itself.
//!
//! The delta cases, from `delta_chain_roundtrips` on, check nothing on a
//! backend whose [`MemoBackend::supports_delta`] is `false`.

use std::sync::{mpsc, Barrier};
use std::thread;
use std::time::Duration;

use super::{
    decode_entry, retag_frame, Chain, DeltaHeader, EncodedEntry, Entry, GenHeader, GenId,
    Generation, MemoBackend, StoredDep, TableDesc, Tombstone,
};
use crate::Error;

/// How long the streamed case waits before calling a write that reads
/// through its own handle deadlocked.
const STREAM_TIMEOUT: Duration = Duration::from_secs(60);

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
            streamed_write_reads_the_base_through_the_same_handle,
            delta_chain_roundtrips,
            tombstone_hides_a_base_entry,
            chain_of_deltas_resolves_newest_first,
            delta_frames_reject_a_bad_tag,
            stale_delta_is_not_returned,
            compaction_drops_the_chain,
            read_entries_matches_read_entry,
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
    tagged_entries(count, 0)
}

fn tagged_entries(count: u8, tag: u64) -> Vec<EncodedEntry> {
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
            EncodedEntry::new(&entry, tag).expect("encode")
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
fn delta_header(base: u64, parent: u64, id: u64) -> DeltaHeader {
    DeltaHeader {
        header: header(id),
        base: GenId(base),
        parent: GenId(parent),
    }
}

/// Entries of a delta: frames tagged with the delta header's tag, keys
/// from `first` on, so a delta may rewrite keys an older layer holds.
fn delta_entries(first: u8, count: u8, header: &DeltaHeader) -> Vec<EncodedEntry> {
    (first..first + count)
        .map(|i| {
            let entry = Entry {
                table: u32::from(i % 2),
                key: vec![i; usize::from(i) + 1],
                value: vec![i ^ 0x5a; usize::from(i) * 3 + 1],
                deps: Vec::new(),
                last_used: header.header.seq,
            };
            EncodedEntry::new(&entry, header.frame_tag()).expect("encode")
        })
        .collect()
}

fn write_delta<B: MemoBackend>(
    backend: &B,
    base: u64,
    tip: u64,
    id: u64,
    written: &[EncodedEntry],
    tombstones: &[Tombstone],
) -> Generation {
    backend
        .write_delta(
            GenId(tip),
            GenId(id),
            &delta_header(base, tip, id),
            written,
            tombstones,
        )
        .expect("write_delta failed")
}

fn chain_ids<B: MemoBackend>(backend: &B) -> Vec<GenId> {
    backend
        .load_chain(&[])
        .expect("load_chain failed")
        .expect("a chain")
        .layers()
        .iter()
        .map(|layer| layer.id())
        .collect()
}

/// A delta written on the current generation comes back from `load_chain`
/// as the tip of a two-layer chain, equal to what `write_delta` returned,
/// with its frames readable. `load` still returns the base, the checkpoint
/// lock keeps both layers, and a new handle sees the same chain.
pub fn delta_chain_roundtrips<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    if !backend.supports_delta() {
        return;
    }
    let base_entries = entries(4);
    let base = write(&backend, None, 1, &base_entries);
    let delta_header = delta_header(1, 1, 2);
    let written = delta_entries(10, 3, &delta_header);
    let delta = write_delta(&backend, 1, 1, 2, &written, &[]);
    assert_eq!(delta.id(), GenId(2));
    assert_eq!(delta.header(), &delta_header.header);
    assert_eq!(delta.base(), Some(GenId(1)));
    assert_eq!(delta.parent(), Some(GenId(1)));
    assert_eq!(delta.frame_tag(), delta_header.frame_tag());
    assert_holds(&backend, &delta, &written);

    drop(backend.lock_checkpoint().expect("lock failed"));
    for handle in [backend, h.open(&store)] {
        let chain = handle
            .load_chain(&[])
            .expect("load_chain failed")
            .expect("a chain");
        assert_eq!(chain.depth(), 1);
        let [loaded_base, loaded_delta] = chain.layers() else {
            panic!("expected two layers, got {:?}", chain.layers());
        };
        assert_eq!(loaded_base.id(), GenId(1));
        assert_eq!(loaded_base.base(), None);
        assert_eq!(loaded_base.items(), base.items());
        assert_holds(&handle, loaded_base, &base_entries);
        assert_eq!(loaded_delta.id(), delta.id());
        assert_eq!(loaded_delta.header(), delta.header());
        assert_eq!(loaded_delta.frame_tag(), delta.frame_tag());
        assert_eq!(loaded_delta.items(), delta.items());
        assert_holds(&handle, loaded_delta, &written);
        let loaded = handle.load().expect("load failed").expect("a generation");
        assert_eq!(loaded.id(), GenId(1), "load returns the base");
    }
}

/// A tombstone in a delta loads as an item of length 0 for its table and
/// key hash, with no frame, so a newest-first probe misses there while the
/// base entry it hides stays readable in the base.
pub fn tombstone_hides_a_base_entry<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    if !backend.supports_delta() {
        return;
    }
    let base_entries = entries(4);
    write(&backend, None, 1, &base_entries);
    let hidden = &base_entries[2];
    let tombstone = Tombstone {
        table: hidden.table,
        key_hash: hidden.key_hash,
    };
    let written = delta_entries(10, 2, &delta_header(1, 1, 2));
    let delta = write_delta(&backend, 1, 1, 2, &written, &[tombstone]);
    for chain in [
        backend.load_chain(&[]).unwrap().unwrap(),
        h.open(&store).load_chain(&[]).unwrap().unwrap(),
    ] {
        let [base, loaded] = chain.layers() else {
            panic!("expected two layers, got {:?}", chain.layers());
        };
        assert_eq!(loaded.items(), delta.items());
        assert_eq!(loaded.items().len(), written.len() + 1);
        let found = loaded.find(tombstone.table, tombstone.key_hash);
        assert_eq!(found.len(), 1);
        assert!(found[0].is_tombstone());
        assert_eq!(
            loaded.entry_bytes(),
            written.iter().map(|e| e.frame.len() as u64).sum::<u64>()
        );
        assert_eq!(
            newest(&backend, &chain, hidden),
            None,
            "the tombstone hides it"
        );
        let item = base.find(hidden.table, hidden.key_hash)[0];
        assert_eq!(
            backend.read_entry(base.id(), &item).unwrap().as_ref(),
            Some(&hidden.frame)
        );
        for entry in &written {
            assert_eq!(newest(&backend, &chain, entry).as_ref(), Some(&entry.frame));
        }
    }
}

/// The frame a newest-first probe of `chain` finds for `entry`'s table
/// and key hash: `None` at a tombstone or when no layer holds it.
fn newest<B: MemoBackend>(backend: &B, chain: &Chain, entry: &EncodedEntry) -> Option<Vec<u8>> {
    for layer in chain.layers().iter().rev() {
        if let Some(item) = layer.find(entry.table, entry.key_hash).first() {
            if item.is_tombstone() {
                return None;
            }
            return backend
                .read_entry(layer.id(), item)
                .expect("read_entry failed");
        }
    }
    None
}

/// Deltas written one on another load oldest first, each naming the base
/// and its predecessor, and a key every layer rewrites resolves to the
/// newest layer's frame when probed newest first.
pub fn chain_of_deltas_resolves_newest_first<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    if !backend.supports_delta() {
        return;
    }
    write(&backend, None, 1, &entries(4));
    let mut last = Vec::new();
    for id in 2..=4 {
        last = delta_entries(1, 2, &delta_header(1, id - 1, id));
        write_delta(&backend, 1, id - 1, id, &last, &[]);
    }
    let chain = h.open(&store).load_chain(&[]).unwrap().unwrap();
    assert_eq!(chain.depth(), 3);
    assert_eq!(chain.base().id(), GenId(1));
    assert_eq!(chain.tip().id(), GenId(4));
    let ids: Vec<_> = chain.layers().iter().map(|l| l.id()).collect();
    assert_eq!(ids, [GenId(1), GenId(2), GenId(3), GenId(4)]);
    for pair in chain.layers().windows(2) {
        assert_eq!(pair[1].base(), Some(GenId(1)));
        assert_eq!(pair[1].parent(), Some(pair[0].id()));
    }
    for entry in &last {
        assert_eq!(newest(&backend, &chain, entry).as_ref(), Some(&entry.frame));
    }
    assert_eq!(backend.load().unwrap().unwrap().id(), GenId(1));
}

/// Every layer's frames decode under its own frame tag only: a delta's tag
/// is its [`DeltaHeader`]'s, not its inner header's, and base frames keep
/// validating against the base header.
pub fn delta_frames_reject_a_bad_tag<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    if !backend.supports_delta() {
        return;
    }
    let base_entries: Vec<_> = (0..3)
        .map(|i| {
            let entry = Entry {
                table: 0,
                key: vec![i],
                value: vec![i; 9],
                deps: Vec::new(),
                last_used: 1,
            };
            EncodedEntry::new(&entry, header(1).frame_tag()).unwrap()
        })
        .collect();
    write(&backend, None, 1, &base_entries);
    let written = delta_entries(10, 2, &delta_header(1, 1, 2));
    write_delta(&backend, 1, 1, 2, &written, &[]);
    let chain = h.open(&store).load_chain(&[]).unwrap().unwrap();
    let [base, delta] = chain.layers() else {
        panic!("expected two layers, got {:?}", chain.layers());
    };
    assert_eq!(base.frame_tag(), header(1).frame_tag());
    assert_eq!(delta.frame_tag(), delta_header(1, 1, 2).frame_tag());
    assert_ne!(delta.frame_tag(), delta.header().frame_tag());
    for (layer, other) in [(base, delta), (delta, base)] {
        for item in layer.items() {
            let frame = backend.read_entry(layer.id(), item).unwrap().unwrap();
            decode_entry(&frame, layer.frame_tag()).expect("a frame decodes under its own tag");
            assert!(matches!(
                decode_entry(&frame, other.frame_tag()),
                Err(Error::Corrupt(_))
            ));
        }
    }
}

/// A delta from a tip that is no longer current is `Busy` and writes
/// nothing visible, and once a full write replaces the chain, `load_chain`
/// returns only the new generation, never a delta of the old base.
pub fn stale_delta_is_not_returned<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    if !backend.supports_delta() {
        return;
    }
    write(&backend, None, 1, &entries(2));
    write_delta(
        &backend,
        1,
        1,
        2,
        &delta_entries(10, 1, &delta_header(1, 1, 2)),
        &[],
    );
    let header = delta_header(1, 1, 3);
    let stale = backend.write_delta(
        GenId(1),
        GenId(3),
        &header,
        &delta_entries(10, 1, &header),
        &[],
    );
    assert!(matches!(stale, Err(Error::Busy)), "got {stale:?}");
    assert_eq!(chain_ids(&backend), [GenId(1), GenId(2)]);

    let full = entries(3);
    write(&backend, Some(GenId(2)), 4, &full);
    let header = delta_header(1, 2, 5);
    let stale = backend.write_delta(
        GenId(2),
        GenId(5),
        &header,
        &delta_entries(10, 1, &header),
        &[],
    );
    assert!(matches!(stale, Err(Error::Busy)), "got {stale:?}");
    for handle in [&backend, &h.open(&store)] {
        let chain = handle.load_chain(&[]).unwrap().unwrap();
        assert_eq!(chain.depth(), 0);
        assert_eq!(chain.tip().id(), GenId(4));
        assert_holds(handle, chain.tip(), &full);
        assert_eq!(handle.load().unwrap().unwrap().id(), GenId(4));
    }
}

/// A full write from the chain's tip replaces the whole chain. Its old
/// layers stay readable until removed or until the checkpoint lock removes
/// them, and racing a delta against a full write from one tip publishes
/// exactly one of them.
pub fn compaction_drops_the_chain<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    if !backend.supports_delta() {
        return;
    }
    let base_entries = entries(3);
    let base = write(&backend, None, 1, &base_entries);
    let written = delta_entries(10, 2, &delta_header(1, 1, 2));
    let delta = write_delta(&backend, 1, 1, 2, &written, &[]);
    let full = entries(5);
    write(&backend, Some(GenId(2)), 3, &full);
    assert_eq!(chain_ids(&backend), [GenId(3)]);
    assert_holds(&backend, &base, &base_entries);
    assert_holds(&backend, &delta, &written);

    backend.remove_generation(GenId(2)).expect("remove failed");
    for item in delta.items() {
        assert_eq!(backend.read_entry(GenId(2), item).unwrap(), None);
    }
    drop(backend.lock_checkpoint().expect("lock failed"));
    for item in base.items() {
        assert_eq!(backend.read_entry(GenId(1), item).unwrap(), None);
    }
    assert_holds(&backend, &backend.load().unwrap().unwrap(), &full);

    let a = h.open(&store);
    let b = h.open(&store);
    let barrier = Barrier::new(2);
    let racing = delta_header(3, 3, 4);
    let delta_written = delta_entries(10, 1, &racing);
    let results = thread::scope(|scope| {
        let delta = scope.spawn(|| {
            barrier.wait();
            a.write_delta(GenId(3), GenId(4), &racing, &delta_written, &[])
                .map(|g| g.id())
        });
        let full = scope.spawn(|| {
            barrier.wait();
            b.write_generation(Some(GenId(3)), GenId(5), &header(5), &entries(2))
                .map(|g| g.id())
        });
        [delta.join().unwrap(), full.join().unwrap()]
    });
    let winners: Vec<_> = results.iter().filter_map(|r| r.as_ref().ok()).collect();
    assert_eq!(winners.len(), 1, "results: {results:?}");
    assert!(results.iter().any(|r| matches!(r, Err(Error::Busy))));
    let tip = h.open(&store).load_chain(&[]).unwrap().unwrap();
    assert_eq!(tip.tip().id(), *winners[0]);
}

/// `read_entries` returns, per item and in order, what `read_entry`
/// returns: the frames of a live layer, and `None` once it is removed.
pub fn read_entries_matches_read_entry<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    if !backend.supports_delta() {
        return;
    }
    let old = write(&backend, None, 1, &entries(5));
    let written = delta_entries(10, 3, &delta_header(1, 1, 2));
    let delta = write_delta(&backend, 1, 1, 2, &written, &[]);
    for layer in [&old, &delta] {
        let mut items = layer.items().to_vec();
        items.reverse();
        let batched = backend.read_entries(layer.id(), &items).unwrap();
        let single: Vec<_> = items
            .iter()
            .map(|item| backend.read_entry(layer.id(), item).unwrap())
            .collect();
        assert_eq!(batched, single);
        assert!(batched.iter().all(Option::is_some));
    }
    assert_eq!(
        backend.read_entries(GenId(2), &[]).unwrap(),
        Vec::<Option<Vec<u8>>>::new()
    );
    write(&backend, Some(GenId(2)), 3, &entries(1));
    backend.remove_generation(GenId(1)).unwrap();
    let gone = backend.read_entries(GenId(1), old.items()).unwrap();
    assert_eq!(gone, vec![None; old.items().len()]);
}

/// A streamed write whose iterator reads the base through the same handle
/// while the write is in progress completes, and a stale base or an
/// iterator error writes nothing visible. Runs on every backend.
pub fn streamed_write_reads_the_base_through_the_same_handle<H: MemoHarness>(mut h: H) {
    let store = h.new_store();
    let backend = h.open(&store);
    let base_entries = tagged_entries(6, header(1).frame_tag());
    let base = write(&backend, None, 1, &base_entries);
    let (done, result) = mpsc::channel();
    let worker = thread::spawn(move || {
        let next = header(2);
        let mut pulls = base.items().iter().map(|item| {
            let frame = backend
                .read_entry(base.id(), item)?
                .expect("the base is current");
            retag_frame(&frame, base.frame_tag(), next.frame_tag())
        });
        let written =
            backend.write_generation_streamed(Some(GenId(1)), GenId(2), &next, &mut pulls);
        let _ = done.send(());
        (backend, written)
    });
    if let Err(mpsc::RecvTimeoutError::Timeout) = result.recv_timeout(STREAM_TIMEOUT) {
        panic!("a streamed write that reads through its own handle did not finish");
    }
    let (backend, written) = worker.join().expect("the streamed write panicked");
    let generation = written.expect("write_generation_streamed failed");
    assert_eq!(generation.id(), GenId(2));
    let loaded = backend.load().unwrap().unwrap();
    assert_eq!(loaded.id(), GenId(2));
    assert_eq!(loaded.items().len(), base_entries.len());
    for entry in &base_entries {
        let item = loaded.find(entry.table, entry.key_hash)[0];
        let frame = backend.read_entry(loaded.id(), &item).unwrap().unwrap();
        let old = decode_entry(&entry.frame, header(1).frame_tag()).unwrap();
        assert_eq!(decode_entry(&frame, header(2).frame_tag()).unwrap(), old);
    }

    let stale = backend.write_generation_streamed(
        Some(GenId(1)),
        GenId(3),
        &header(3),
        &mut entries(2).into_iter().map(Ok),
    );
    assert!(matches!(stale, Err(Error::Busy)), "got {stale:?}");
    let mut failing = entries(2)
        .into_iter()
        .map(Ok)
        .chain(std::iter::once(Err(Error::Corrupt("pull failed"))));
    let failed =
        backend.write_generation_streamed(Some(GenId(2)), GenId(4), &header(4), &mut failing);
    assert!(
        matches!(failed, Err(Error::Corrupt("pull failed"))),
        "got {failed:?}"
    );
    let loaded = h.open(&store).load().unwrap().unwrap();
    assert_eq!(loaded.id(), GenId(2));
    assert_eq!(loaded.items().len(), base_entries.len());
}
