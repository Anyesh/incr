//! Memo state a restore can't vouch for: damaged, swapped, rolled back,
//! lossy or crafted generations, mapped writes that panicked and writes
//! around the store. Every such entry runs cold, nothing over it is kept,
//! and values always equal a cold run's.

use std::cell::Cell;
use std::fs;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_core::persist_hooks::assert_invariants;
use incr_persist::memo::{
    decode_entry, fingerprint, BuildId, EncodedEntry, Entry, FileMemo, GenId, Memo, MemoBackend,
    MemoBuilder, MemoCells, MemoOptions, MemoTable, MemoryMemo, StoredDep,
};
use incr_persist::{Input, Local, MemoryBackend, Options, Shared, Store};
use serde::{Deserialize, Serialize, Serializer};

#[path = "../../incr-core/tests/common/child.rs"]
mod child;

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

both!(
    a_lossy_value_is_not_kept,
    a_key_that_decodes_to_another_runs_cold,
    a_panicking_eq_in_a_mapped_write_untrusts_the_input,
    a_bypass_restored_and_checkpointed_keeps_nothing_over_it,
    an_equal_mapped_write_does_not_launder_a_bypass,
    a_bypass_before_open_is_not_trusted,
    two_restores_in_one_compute_run_once,
    recomputed_seeded_dep_invalidates_trusting_entry,
    another_key_type_drops_only_that_table,
);

/// On real files, and the flip and truncation tests restart twice per
/// byte, so these run outside Miri; each runs Local, then Shared.
mod files {
    macro_rules! file_tests {
        ($($name:ident),* $(,)?) => {
            $(
                #[test]
                #[cfg_attr(miri, ignore)]
                fn $name() {
                    super::$name::<super::Local>();
                    super::$name::<super::Shared>();
                }
            )*
        };
    }

    file_tests!(
        every_byte_flip_in_a_generation_file_is_caught,
        every_truncation_of_a_generation_file_runs_cold,
        swapped_generation_files_are_not_served,
    );

    #[cfg(feature = "sqlite")]
    file_tests!(a_lost_buffered_sqlite_generation_rolls_back_whole);
}

const BUILD: u128 = 0xc0aa;

type St<C> = Arc<Store<C, MemoryBackend>>;

fn options() -> MemoOptions {
    MemoOptions::new(BuildId::Fixed(BUILD))
}

fn store<C: MemoCells>(log: &MemoryBackend) -> St<C> {
    Arc::new(Store::open(log.clone(), Options::default()).unwrap())
}

fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

/// `chain(0) = x` and `chain(k) = chain(k - 1) + k`, counting runs.
struct Chain<C: MemoCells> {
    store: St<C>,
    x: Input<u64>,
    memo: Memo<C, MemoryBackend>,
    chain: MemoTable<u32, u64>,
}

fn cold_chain(x: u64, k: u32) -> u64 {
    x + u64::from(k * (k + 1) / 2)
}

impl<C: MemoCells> Chain<C> {
    fn open(
        log: &MemoryBackend,
        backend: impl MemoBackend + 'static,
        runs: &Arc<AtomicUsize>,
    ) -> Self {
        let store = store::<C>(log);
        let x = store.input("x", 1_u64).unwrap();
        Chain::on(store, x, backend, runs)
    }

    fn on(
        store: St<C>,
        x: Input<u64>,
        backend: impl MemoBackend + 'static,
        runs: &Arc<AtomicUsize>,
    ) -> Self {
        let xh = x.handle();
        let mut builder = MemoBuilder::new(options());
        let chain = builder.table::<u32, u64>("chain", 1);
        let runs = Arc::clone(runs);
        builder.define(chain, move |ctx, &k| {
            runs.fetch_add(1, SeqCst);
            if k == 0 {
                ctx.rt().get(xh)
            } else {
                ctx.get(chain, k - 1) + u64::from(k)
            }
        });
        let memo = builder.open(&store, backend).unwrap();
        Chain {
            store,
            x,
            memo,
            chain,
        }
    }

    fn read(&self, k: u32) -> u64 {
        let q = self.memo.get_or_insert(self.chain, k);
        self.memo.runtime().get(q)
    }

    fn x(&self) -> u64 {
        self.memo.runtime().get(self.x.handle())
    }

    /// Reads `chain(top)` first, so one restore walks the whole chain,
    /// then checks every key against a cold run and the runtime's
    /// invariants.
    fn check(&self, top: u32) {
        let x = self.x();
        assert_eq!(self.read(top), cold_chain(x, top));
        for k in 0..top {
            assert_eq!(self.read(k), cold_chain(x, k), "chain({k})");
        }
        assert_invariants(self.memo.runtime());
    }
}

const TOP: u32 = 4;
const ENTRIES: u64 = TOP as u64 + 1;
const GEN_1: &str = "MEMO_GEN.1";

/// Writes `chain(0..=TOP)` over x = 3 as generation 1 of a `FileMemo` in
/// `dir`, and returns every file the directory then holds.
fn file_generation<C: MemoCells>(dir: &Path, log: &MemoryBackend) -> Vec<(PathBuf, Vec<u8>)> {
    let w = Chain::<C>::open(log, FileMemo::open(dir).unwrap(), &counter());
    w.store.set(w.x, 3).unwrap();
    w.check(TOP);
    assert_eq!(w.memo.checkpoint().unwrap().written, ENTRIES as usize);
    drop(w);
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let path = e.unwrap().path();
            let bytes = fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect();
    files.sort();
    files
}

fn reset_dir(dir: &Path, files: &[(PathBuf, Vec<u8>)]) {
    for entry in fs::read_dir(dir).unwrap() {
        fs::remove_file(entry.unwrap().path()).unwrap();
    }
    for (path, bytes) in files {
        fs::write(path, bytes).unwrap();
    }
}

/// After damage, a checkpoint writes every live entry again and the next
/// restart reuses them all without running anything.
fn assert_recovers<C: MemoCells>(
    w: Chain<C>,
    log: &MemoryBackend,
    dir: &Path,
    runs: &Arc<AtomicUsize>,
) {
    assert_eq!(w.memo.checkpoint().unwrap().written, ENTRIES as usize);
    drop(w);
    let w = Chain::<C>::open(log, FileMemo::open(dir).unwrap(), runs);
    let before = runs.load(SeqCst);
    w.check(TOP);
    assert_eq!(runs.load(SeqCst), before);
    assert_eq!(w.memo.stats().reused, ENTRIES);
}

fn every_byte_flip_in_a_generation_file_is_caught<C: MemoCells>() {
    let dir = tempfile::tempdir().unwrap();
    let log = MemoryBackend::new();
    let pristine = file_generation::<C>(dir.path(), &log);
    let gen_path = dir.path().join(GEN_1);
    let generation = FileMemo::open(dir.path()).unwrap().load().unwrap().unwrap();
    let in_frame = |at: u64| {
        generation
            .items()
            .iter()
            .any(|item| (item.loc..item.loc + u64::from(item.len)).contains(&at))
    };
    let bytes = fs::read(&gen_path).unwrap();
    let runs = counter();
    for at in 0..bytes.len() {
        reset_dir(dir.path(), &pristine);
        let mut damaged = bytes.clone();
        damaged[at] ^= 0x10;
        fs::write(&gen_path, &damaged).unwrap();
        let w = Chain::<C>::open(&log, FileMemo::open(dir.path()).unwrap(), &runs);
        w.check(TOP);
        let stats = w.memo.stats();
        assert!(stats.reused < ENTRIES, "the flip at {at} went unnoticed");
        if in_frame(at as u64) {
            assert!(stats.corrupt >= 1, "the flip at {at} wasn't counted");
        }
        assert_recovers(w, &log, dir.path(), &runs);
    }
}

fn every_truncation_of_a_generation_file_runs_cold<C: MemoCells>() {
    let dir = tempfile::tempdir().unwrap();
    let log = MemoryBackend::new();
    let pristine = file_generation::<C>(dir.path(), &log);
    let gen_path = dir.path().join(GEN_1);
    let bytes = fs::read(&gen_path).unwrap();
    let runs = counter();
    for len in 0..bytes.len() {
        reset_dir(dir.path(), &pristine);
        fs::write(&gen_path, &bytes[..len]).unwrap();
        let w = Chain::<C>::open(&log, FileMemo::open(dir.path()).unwrap(), &runs);
        w.check(TOP);
        let stats = w.memo.stats();
        assert_eq!(
            (stats.reused, stats.revalidated),
            (0, 0),
            "truncated to {len}"
        );
        assert_recovers(w, &log, dir.path(), &runs);
    }
}

/// `a(k) = k * x` and `b(k) = k * x + 1`, declared in either order.
struct Pair<C: MemoCells> {
    _store: St<C>,
    memo: Memo<C, MemoryBackend>,
    a: MemoTable<u32, u64>,
    b: MemoTable<u32, u64>,
}

impl<C: MemoCells> Pair<C> {
    fn open(log: &MemoryBackend, dir: &Path, b_first: bool, runs: &Arc<AtomicUsize>) -> Self {
        let store = store::<C>(log);
        let x = store.input("x", 1_u64).unwrap().handle();
        let mut builder = MemoBuilder::new(options());
        let (a, b) = if b_first {
            let b = builder.table::<u32, u64>("b", 1);
            (builder.table::<u32, u64>("a", 1), b)
        } else {
            let a = builder.table::<u32, u64>("a", 1);
            (a, builder.table::<u32, u64>("b", 1))
        };
        let r = Arc::clone(runs);
        builder.define(a, move |ctx, &k| {
            r.fetch_add(1, SeqCst);
            u64::from(k) * ctx.rt().get(x)
        });
        let r = Arc::clone(runs);
        builder.define(b, move |ctx, &k| {
            r.fetch_add(1, SeqCst);
            u64::from(k) * ctx.rt().get(x) + 1
        });
        let memo = builder.open(&store, FileMemo::open(dir).unwrap()).unwrap();
        Pair {
            _store: store,
            memo,
            a,
            b,
        }
    }

    fn check(&self) {
        let rt = self.memo.runtime();
        assert_eq!(rt.get(self.memo.get_or_insert(self.b, 1)), 2);
        assert_eq!(rt.get(self.memo.get_or_insert(self.a, 1)), 1);
        assert_invariants(rt);
    }
}

fn swap(dir: &Path, one: &str, two: &str) {
    let (one, two, tmp) = (dir.join(one), dir.join(two), dir.join("swap"));
    fs::rename(&one, &tmp).unwrap();
    fs::rename(&two, &one).unwrap();
    fs::rename(&tmp, &two).unwrap();
}

/// Generation 1 catalogs `[a, b]` and generation 2 `[b, a]` over the same
/// x, so the frame at `b(1)`'s offset in generation 2's index is, in
/// generation 1's file, a whole frame of `a(1)` under the same table
/// index and key. Generation 2's index must never read generation 1's
/// file, whichever name it carries.
fn swapped_generation_files_are_not_served<C: MemoCells>() {
    let dir = tempfile::tempdir().unwrap();
    let log = MemoryBackend::new();
    let runs = counter();
    {
        let w = Pair::<C>::open(&log, dir.path(), false, &runs);
        w.check();
        assert_eq!(w.memo.checkpoint().unwrap().written, 2);
    }
    let first = fs::read(dir.path().join(GEN_1)).unwrap();
    {
        let w = Pair::<C>::open(&log, dir.path(), true, &runs);
        w.check();
        assert_eq!(w.memo.stats().reused, 2);
        assert_eq!(w.memo.checkpoint().unwrap().written, 2);
    }
    assert!(!dir.path().join(GEN_1).exists());
    let second = FileMemo::open(dir.path()).unwrap().load().unwrap().unwrap();
    let b_item = second
        .items()
        .iter()
        .find(|item| item.table == 0)
        .copied()
        .unwrap();
    let at = b_item.loc as usize..b_item.loc as usize + b_item.len as usize;
    let misread = decode_entry(&first[at]).unwrap();
    assert_eq!((misread.table, misread.key.as_slice()), (0, &[1_u8][..]));

    // Swapped after this process loaded generation 2: it keeps reading the
    // file it loaded.
    let w = Pair::<C>::open(&log, dir.path(), true, &runs);
    fs::write(dir.path().join(GEN_1), &first).unwrap();
    swap(dir.path(), GEN_1, "MEMO_GEN.2");
    let before = runs.load(SeqCst);
    w.check();
    assert_eq!(runs.load(SeqCst), before);
    drop(w);

    // Swapped before: the file named generation 2 holds generation 1.
    let w = Pair::<C>::open(&log, dir.path(), true, &runs);
    let before = runs.load(SeqCst);
    w.check();
    assert_eq!(runs.load(SeqCst), before + 2);
    let stats = w.memo.stats();
    assert_eq!((stats.reused, stats.revalidated), (0, 0));
    assert_eq!(w.memo.checkpoint().unwrap().written, 2);
    drop(w);
    let w = Pair::<C>::open(&log, dir.path(), true, &runs);
    w.check();
    assert_eq!(w.memo.stats().reused, 2);
}

/// Under `Buffered`, a crash can lose the latest generation, which SQLite
/// holds only in its write-ahead log until it checkpoints. The memo falls
/// back a generation while the log keeps its later writes.
#[cfg(feature = "sqlite")]
fn a_lost_buffered_sqlite_generation_rolls_back_whole<C: MemoCells>() {
    use incr_persist::memo::{SqliteMemo, SqliteMemoOptions};
    use incr_persist::SqliteDurability;

    let dir = tempfile::tempdir().unwrap();
    let (path, crashed) = (dir.path().join("memo.sqlite"), dir.path().join("crashed"));
    let buffered = || {
        let options = SqliteMemoOptions {
            durability: SqliteDurability::Buffered,
            ..SqliteMemoOptions::default()
        };
        SqliteMemo::open_with(&path, options).unwrap()
    };
    let log = MemoryBackend::new();
    let runs = counter();
    {
        let w = Chain::<C>::open(&log, buffered(), &runs);
        w.store.set(w.x, 3).unwrap();
        w.check(TOP);
        w.memo.checkpoint().unwrap();
    }
    {
        let w = Chain::<C>::open(&log, buffered(), &runs);
        w.store.set(w.x, 5).unwrap();
        w.check(TOP + 2);
        w.memo.checkpoint().unwrap();
        // The main database file without its write-ahead log: what a power
        // loss before SQLite's next checkpoint leaves.
        fs::copy(&path, &crashed).unwrap();
    }
    let survived = SqliteMemo::open(&crashed).unwrap().load().unwrap().unwrap();
    assert_eq!(survived.id(), GenId(1));
    assert_eq!(survived.items().len(), ENTRIES as usize);

    let w = Chain::<C>::open(&log, SqliteMemo::open(&crashed).unwrap(), &runs);
    assert_eq!(w.x(), 5);
    let before = runs.load(SeqCst);
    w.check(TOP + 2);
    assert!(
        runs.load(SeqCst) > before,
        "nothing over x = 3 may be reused"
    );
    let stats = w.memo.stats();
    assert_eq!((stats.corrupt, stats.cold), (0, 0));
    assert_eq!(w.memo.checkpoint().unwrap().written, ENTRIES as usize + 2);
    drop(w);
    let w = Chain::<C>::open(&log, SqliteMemo::open(&crashed).unwrap(), &runs);
    let before = runs.load(SeqCst);
    w.check(TOP + 2);
    assert_eq!(runs.load(SeqCst), before);
}

/// `serde(skip)` drops `b`, so a value with `b != 0` decodes unequal.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Lossy {
    a: u64,
    #[serde(skip)]
    b: u64,
}

/// `lossy(k) = {a: x, b: k}` and `sum(k) = lossy(k).a + lossy(k).b`.
fn a_lossy_value_is_not_kept<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let runs = counter();
    let open = || {
        let store = store::<C>(&log);
        let x = store.input("x", 7_u64).unwrap().handle();
        let mut builder = MemoBuilder::new(options());
        let lossy = builder.table::<u32, Lossy>("lossy", 1);
        let sum = builder.table::<u32, u64>("sum", 1);
        let r = Arc::clone(&runs);
        builder.define(lossy, move |ctx, &k| {
            r.fetch_add(1, SeqCst);
            Lossy {
                a: ctx.rt().get(x),
                b: u64::from(k),
            }
        });
        let r = Arc::clone(&runs);
        builder.define(sum, move |ctx, &k| {
            r.fetch_add(1, SeqCst);
            let v = ctx.get(lossy, k);
            v.a + v.b
        });
        let memo = builder.open(&store, backend.clone()).unwrap();
        (store, memo, lossy, sum)
    };
    {
        let (_store, memo, _lossy, sum) = open();
        let rt = memo.runtime();
        assert_eq!(rt.get(memo.get_or_insert(sum, 0)), 7);
        assert_eq!(rt.get(memo.get_or_insert(sum, 1)), 8);
        assert_eq!(memo.checkpoint().unwrap().written, 2);
    }
    let (_store, memo, lossy, sum) = open();
    let rt = memo.runtime();
    let before = runs.load(SeqCst);
    assert_eq!(rt.get(memo.get_or_insert(sum, 1)), 8);
    assert_eq!(rt.get(memo.get_or_insert(lossy, 1)), Lossy { a: 7, b: 1 });
    assert_eq!(runs.load(SeqCst), before + 2);
    assert_eq!(rt.get(memo.get_or_insert(sum, 0)), 7);
    assert_eq!(rt.get(memo.get_or_insert(lossy, 0)), Lossy { a: 7, b: 0 });
    assert_eq!(runs.load(SeqCst), before + 2);
    assert_eq!(memo.stats().reused, 2);
    assert_invariants(rt);
}

/// `{a: 1, b: 5}` encodes to the bytes of `{a: 1, b: 0}`, whose entry is
/// stored, so its restore finds that entry and must refuse it.
fn a_key_that_decodes_to_another_runs_cold<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let runs = counter();
    let open = || {
        let store = store::<C>(&log);
        let x = store.input("x", 1000_u64).unwrap().handle();
        let mut builder = MemoBuilder::new(options());
        let t = builder.table::<Lossy, u64>("t", 1);
        let r = Arc::clone(&runs);
        builder.define(t, move |ctx, k: &Lossy| {
            r.fetch_add(1, SeqCst);
            ctx.rt().get(x) + k.a * 10 + k.b
        });
        let memo = builder.open(&store, backend.clone()).unwrap();
        (store, memo, t)
    };
    let (kept, lossy) = (Lossy { a: 1, b: 0 }, Lossy { a: 1, b: 5 });
    {
        let (_store, memo, t) = open();
        let rt = memo.runtime();
        assert_eq!(rt.get(memo.get_or_insert(t, kept.clone())), 1010);
        assert_eq!(rt.get(memo.get_or_insert(t, lossy.clone())), 1015);
        assert_eq!(memo.checkpoint().unwrap().written, 1);
    }
    let (_store, memo, t) = open();
    let rt = memo.runtime();
    let before = runs.load(SeqCst);
    assert_eq!(rt.get(memo.get_or_insert(t, lossy)), 1015);
    assert_eq!(runs.load(SeqCst), before + 1);
    let stats = memo.stats();
    assert_eq!((stats.reused, stats.cold), (0, 1));
    assert_eq!(rt.get(memo.get_or_insert(t, kept)), 1010);
    assert_eq!(runs.load(SeqCst), before + 1);
    assert_eq!(memo.stats().reused, 1);
    assert_invariants(rt);
}

/// A generation written straight through the backend whose query deps
/// form cycles: a self-loop at `chain(1)`, `chain(2)` and `chain(3)` reading
/// each other, and `chain(4)` over `chain(5)` and `chain(6)`, which read
/// each other. Every stored value is a bogus 999 whose fingerprint the
/// deps record. Each root restores in a fresh process; the walk must end,
/// and the values must be the closure's.
fn crafted_cyclic_entries_restore_cold<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let runs = counter();
    {
        let w = Chain::<C>::open(&log, backend.clone(), &runs);
        w.store.set(w.x, 3).unwrap();
        w.check(0);
        w.memo.checkpoint().unwrap();
    }
    let first = backend.load().unwrap().unwrap();
    let real = decode_entry(
        &backend
            .read_entry(first.id(), &first.items()[0])
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    let x_dep = real.deps[0].clone();
    assert!(matches!(x_dep, StoredDep::Input { .. }));
    let key = |k: u32| postcard::to_allocvec(&k).unwrap();
    let bogus = postcard::to_allocvec(&999_u64).unwrap();
    let query = |k: u32| StoredDep::Query {
        table: real.table,
        key: key(k),
        fp: fingerprint(&bogus),
    };
    let entry = |k: u32, deps: Vec<StoredDep>| Entry {
        table: real.table,
        key: key(k),
        value: bogus.clone(),
        deps,
        last_used: 2,
    };
    let crafted = [
        real.clone(),
        entry(1, vec![query(1)]),
        entry(2, vec![query(3)]),
        entry(3, vec![query(2)]),
        entry(4, vec![query(5), x_dep]),
        entry(5, vec![query(6)]),
        entry(6, vec![query(5)]),
    ];
    let mut header = first.header().clone();
    header.seq = 2;
    let encoded: Vec<_> = crafted
        .iter()
        .map(|e| EncodedEntry::new(e).unwrap())
        .collect();
    backend
        .write_generation(Some(first.id()), GenId(2), &header, &encoded)
        .unwrap();
    for root in 1..=6 {
        let w = Chain::<C>::open(&log, backend.clone(), &runs);
        w.check(root);
        w.check(6);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn crafted_cyclic_entries_restore_cold_local() {
    child::run_in_child(
        "crafted_cyclic_entries_restore_cold_local",
        crafted_cyclic_entries_restore_cold::<Local>,
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn crafted_cyclic_entries_restore_cold_shared() {
    child::run_in_child(
        "crafted_cyclic_entries_restore_cold_shared",
        crafted_cyclic_entries_restore_cold::<Shared>,
    )
    .assert_clean();
}

/// Which user call panics while armed: `Eq` and `Hash` on a POISON value,
/// `Serialize` on its second encoding (the first is the log record's,
/// before the mapped write).
#[derive(Clone, Copy)]
enum Trap {
    Off,
    Eq,
    Hash,
    Serialize { skip: u32 },
}

const POISON: u64 = 7_777;

thread_local! {
    static TRAP: Cell<Trap> = const { Cell::new(Trap::Off) };
}

fn trap() -> Trap {
    TRAP.with(Cell::get)
}

/// Runs `write` with `trap` armed and asserts that the trap is what made
/// it panic.
fn trip(trap: Trap, write: impl FnOnce()) {
    TRAP.with(|t| t.set(trap));
    let result = catch_unwind(AssertUnwindSafe(write));
    TRAP.with(|t| t.set(Trap::Off));
    let payload = result.expect_err("the armed write did not panic");
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(message.contains("trap"), "another panic: {message}");
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Touchy(u64);

impl PartialEq for Touchy {
    fn eq(&self, other: &Self) -> bool {
        if matches!(trap(), Trap::Eq) && (self.0 == POISON || other.0 == POISON) {
            panic!("Touchy::eq trap");
        }
        self.0 == other.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
struct Elem(u64);

impl Hash for Elem {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if matches!(trap(), Trap::Hash) && self.0 == POISON {
            panic!("Elem::hash trap");
        }
        self.0.hash(state);
    }
}

impl Serialize for Elem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Trap::Serialize { skip } = trap() {
            if self.0 == POISON {
                assert!(skip > 0, "Elem::serialize trap");
                TRAP.with(|t| t.set(Trap::Serialize { skip: skip - 1 }));
            }
        }
        serializer.serialize_newtype_struct("Elem", &self.0)
    }
}

struct Probe<'a, C: MemoCells> {
    memo: &'a Memo<C, MemoryBackend>,
    table: MemoTable<u32, u64>,
}

impl<C: MemoCells> Probe<'_, C> {
    fn read(&self, k: u32) -> u64 {
        self.memo
            .runtime()
            .get(self.memo.get_or_insert(self.table, k))
    }
}

/// The body of each mapped-write panic test, over a memo whose table
/// computes `t(k) = k * dep` for its one store dep. Stores `t(1)`, `t(2)`
/// and `t(5)`; after a restart restores `t(1)` and runs `panic`, a mapped
/// write of the dep that panics. Then `t(2)`, restored after the panic,
/// `t(3)`, new, and `t(5)`, restored after a checkpoint, all run cold; the
/// checkpoint writes only the stored entries it merges; and the next
/// restart finds no entry it didn't store before the panic.
fn assert_untrusted_after_panic<C: MemoCells, W>(
    open: impl Fn() -> (W, Memo<C, MemoryBackend>, MemoTable<u32, u64>),
    dep: impl Fn(&W, &Memo<C, MemoryBackend>) -> u64,
    panic: impl FnOnce(&W),
) {
    {
        let (_w, memo, table) = open();
        let p = Probe { memo: &memo, table };
        let d = dep(&_w, &memo);
        for k in [1, 2, 5] {
            assert_eq!(p.read(k), u64::from(k) * d);
        }
        assert_eq!(memo.checkpoint().unwrap().written, 3);
    }
    {
        let (w, memo, table) = open();
        let p = Probe { memo: &memo, table };
        let d = dep(&w, &memo);
        assert_eq!(p.read(1), d);
        assert_eq!(memo.stats().reused, 1);
        panic(&w);
        let d = dep(&w, &memo);
        assert_eq!(p.read(2), 2 * d);
        assert_eq!(p.read(3), 3 * d);
        let stats = memo.checkpoint().unwrap();
        assert_eq!((stats.written, stats.merged), (3, 3));
        assert_eq!(p.read(5), 5 * d);
        let stats = memo.stats();
        assert_eq!((stats.reused, stats.revalidated, stats.cold), (1, 0, 2));
        assert_invariants(memo.runtime());
    }
    let (w, memo, table) = open();
    let p = Probe { memo: &memo, table };
    let d = dep(&w, &memo);
    for k in [1, 2, 3, 5] {
        assert_eq!(p.read(k), u64::from(k) * d);
    }
    let stats = memo.stats();
    assert_eq!(stats.reused + stats.revalidated, 3, "t(3) was kept");
    assert_invariants(memo.runtime());
}

fn a_panicking_eq_in_a_mapped_write_untrusts_the_input<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let open = || {
        let store = store::<C>(&log);
        let x = store.input("x", Touchy(2)).unwrap();
        let xh = x.handle();
        let mut builder = MemoBuilder::new(options());
        let t = builder.table::<u32, u64>("t", 1);
        builder.define(t, move |ctx, &k| u64::from(k) * ctx.rt().get(xh).0);
        let memo = builder.open(&store, backend.clone()).unwrap();
        ((store, x), memo, t)
    };
    assert_untrusted_after_panic::<C, _>(
        open,
        |(_, x), memo| memo.runtime().get(x.handle()).0,
        |(store, x)| {
            trip(Trap::Eq, || {
                let _ = store.set(*x, Touchy(POISON));
            })
        },
    );
}

type Items = incr_persist::PersistedCollection<Elem, Shared>;

fn collection_panic(trap: Trap) {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let fresh = Cell::new(true);
    let open = || {
        let store = store::<Shared>(&log);
        let items: Items = store.collection("items").unwrap();
        if fresh.replace(false) {
            items.insert(&store, Elem(4)).unwrap();
            items.insert(&store, Elem(5)).unwrap();
        }
        let view = items.handle().clone();
        let mut builder = MemoBuilder::new(options());
        let t = builder.table::<u32, u64>("t", 1);
        builder.define(t, move |ctx, &k| {
            u64::from(k) * view.values(ctx.rt()).iter().map(|e| e.0).sum::<u64>()
        });
        let memo = builder.open(&store, backend.clone()).unwrap();
        ((store, items), memo, t)
    };
    assert_untrusted_after_panic::<Shared, _>(
        open,
        |(_, items), memo| {
            items
                .handle()
                .values(memo.runtime())
                .iter()
                .map(|e| e.0)
                .sum()
        },
        |(store, items)| {
            trip(trap, || {
                let _ = items.insert(store, Elem(POISON));
            })
        },
    );
}

#[test]
fn a_panicking_hash_in_a_mapped_insert_untrusts_the_collection() {
    collection_panic(Trap::Hash);
}

#[test]
fn a_panicking_serialize_in_a_mapped_insert_untrusts_the_collection() {
    collection_panic(Trap::Serialize { skip: 1 });
}

/// A write around the store, then a restore over it and a checkpoint:
/// the restart, whose log never saw the write, gets the logged value's
/// results and none computed from the bypassed one.
fn a_bypass_restored_and_checkpointed_keeps_nothing_over_it<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let runs = counter();
    {
        let w = Chain::<C>::open(&log, backend.clone(), &runs);
        w.check(2);
        w.memo.checkpoint().unwrap();
    }
    {
        let w = Chain::<C>::open(&log, backend.clone(), &runs);
        w.memo.runtime().set(w.x.handle(), 5);
        let before = runs.load(SeqCst);
        w.check(2);
        assert_eq!(runs.load(SeqCst), before + 3, "an entry over x verified");
        let stats = w.memo.checkpoint().unwrap();
        assert_eq!((stats.written, stats.merged), (3, 3));
    }
    let w = Chain::<C>::open(&log, backend.clone(), &runs);
    assert_eq!(w.x(), 1);
    let before = runs.load(SeqCst);
    w.check(2);
    assert_eq!(runs.load(SeqCst), before);
}

/// After a bypass to 5, another store logs x = 5 and this one applies it
/// on refresh. The runtime already holds 5, so the mapped write is a
/// no-op and leaves the mapped stamp behind: the stored entries over
/// x = 1 still must not verify.
fn an_equal_mapped_write_does_not_launder_a_bypass<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let runs = counter();
    {
        let w = Chain::<C>::open(&log, backend.clone(), &runs);
        w.check(2);
        w.memo.checkpoint().unwrap();
    }
    {
        let w = Chain::<C>::open(&log, backend.clone(), &runs);
        w.memo.runtime().set(w.x.handle(), 5);
        let other = store::<C>(&log);
        let x = other.input("x", 1_u64).unwrap();
        other.set(x, 5).unwrap();
        w.store.refresh().unwrap();
        assert_eq!(w.x(), 5);
        let before = runs.load(SeqCst);
        w.check(2);
        assert_eq!(runs.load(SeqCst), before + 3, "an entry over x verified");
        let stats = w.memo.checkpoint().unwrap();
        assert_eq!((stats.written, stats.merged), (3, 3));
    }
    let w = Chain::<C>::open(&log, backend.clone(), &runs);
    assert_eq!(w.x(), 5);
    w.check(2);
}

fn a_bypass_before_open_is_not_trusted<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let runs = counter();
    {
        let w = Chain::<C>::open(&log, backend.clone(), &runs);
        w.check(2);
        w.memo.checkpoint().unwrap();
    }
    {
        let store = store::<C>(&log);
        let x = store.input("x", 1_u64).unwrap();
        store.runtime().set(x.handle(), 5);
        let w = Chain::on(store, x, backend.clone(), &runs);
        let before = runs.load(SeqCst);
        w.check(2);
        assert_eq!(runs.load(SeqCst), before + 3, "an entry over x verified");
        let stats = w.memo.checkpoint().unwrap();
        assert_eq!((stats.written, stats.merged), (3, 3));
    }
    let w = Chain::<C>::open(&log, backend.clone(), &runs);
    let before = runs.load(SeqCst);
    w.check(2);
    assert_eq!(runs.load(SeqCst), before);
}

/// Only `a` and `b`, over `a`, were checkpointed. `p` runs cold and
/// restores `a` and then `b` from inside its compute, two restores the
/// second of which trusts the first's seed: neither runs, and `p` runs
/// once.
fn two_restores_in_one_compute_run_once<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let leaf_runs = counter();
    let p_runs = counter();
    let open = || {
        let store = store::<C>(&log);
        let x = store.input("x", 2_u64).unwrap().handle();
        let mut builder = MemoBuilder::new(options());
        let a = builder.table::<u32, u64>("a", 1);
        let b = builder.table::<u32, u64>("b", 1);
        let p = builder.table::<u32, u64>("p", 1);
        let r = Arc::clone(&leaf_runs);
        builder.define(a, move |ctx, _| {
            r.fetch_add(1, SeqCst);
            ctx.rt().get(x) * 10
        });
        let r = Arc::clone(&leaf_runs);
        builder.define(b, move |ctx, &k| {
            r.fetch_add(1, SeqCst);
            ctx.get(a, k) + 1
        });
        let r = Arc::clone(&p_runs);
        builder.define(p, move |ctx, &k| {
            r.fetch_add(1, SeqCst);
            ctx.get(a, k) + ctx.get(b, k)
        });
        let memo = builder.open(&store, backend.clone()).unwrap();
        (store, memo, b, p)
    };
    {
        let (_store, memo, b, _p) = open();
        assert_eq!(memo.runtime().get(memo.get_or_insert(b, 0)), 21);
        assert_eq!(memo.checkpoint().unwrap().written, 2);
    }
    let (_store, memo, _b, p) = open();
    assert_eq!(memo.runtime().get(memo.get_or_insert(p, 0)), 41);
    assert_eq!(p_runs.load(SeqCst), 1);
    assert_eq!(leaf_runs.load(SeqCst), 2);
    assert_eq!(memo.stats().reused, 2);
    assert_invariants(memo.runtime());
}

/// `r` trusts `a` at its seed stamp. `a`, over a changed x, runs first
/// and changes its value, so `r` must run although its own deps verified.
fn recomputed_seeded_dep_invalidates_trusting_entry<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let a_runs = counter();
    let r_runs = counter();
    let open = || {
        let store = store::<C>(&log);
        let x = store.input("x", 1_u64).unwrap();
        let xh = x.handle();
        let mut builder = MemoBuilder::new(options());
        let a = builder.table::<u32, u64>("a", 1);
        let r = builder.table::<u32, u64>("r", 1);
        let runs = Arc::clone(&a_runs);
        builder.define(a, move |ctx, _| {
            runs.fetch_add(1, SeqCst);
            ctx.rt().get(xh) * 2
        });
        let runs = Arc::clone(&r_runs);
        builder.define(r, move |ctx, &k| {
            runs.fetch_add(1, SeqCst);
            ctx.get(a, k) + 1
        });
        let memo = builder.open(&store, backend.clone()).unwrap();
        (store, x, memo, r)
    };
    {
        let (_store, _x, memo, r) = open();
        assert_eq!(memo.runtime().get(memo.get_or_insert(r, 0)), 3);
        assert_eq!(memo.checkpoint().unwrap().written, 2);
    }
    let (store, x, memo, r) = open();
    store.set(x, 4).unwrap();
    assert_eq!(memo.runtime().get(memo.get_or_insert(r, 0)), 9);
    assert_eq!((a_runs.load(SeqCst), r_runs.load(SeqCst)), (2, 2));
    let stats = memo.stats();
    assert_eq!((stats.reused, stats.revalidated), (1, 1));
    assert_invariants(memo.runtime());
}

/// `scaled` changes its key type from u32 to u64, whose postcard bytes
/// for 2 are the same, and its closure with it. Its entry must go, and
/// `other`'s must stay.
fn another_key_type_drops_only_that_table<C: MemoCells>() {
    let log = MemoryBackend::new();
    let backend = MemoryMemo::new();
    let runs = counter();
    let open = |wide: bool| {
        let store = store::<C>(&log);
        let x = store.input("x", 3_u64).unwrap().handle();
        let mut builder = MemoBuilder::new(options());
        let other = builder.table::<u32, u64>("other", 1);
        let r = Arc::clone(&runs);
        builder.define(other, move |ctx, &k| {
            r.fetch_add(1, SeqCst);
            ctx.rt().get(x) + u64::from(k)
        });
        let r = Arc::clone(&runs);
        let read = if wide {
            let scaled = builder.table::<u64, u64>("scaled", 1);
            builder.define(scaled, move |ctx, &k| {
                r.fetch_add(1, SeqCst);
                ctx.rt().get(x) * k * 1000
            });
            Box::new(move |memo: &Memo<C, MemoryBackend>| {
                memo.runtime().get(memo.get_or_insert(scaled, 2))
            }) as Box<dyn Fn(&Memo<C, MemoryBackend>) -> u64>
        } else {
            let scaled = builder.table::<u32, u64>("scaled", 1);
            builder.define(scaled, move |ctx, &k| {
                r.fetch_add(1, SeqCst);
                ctx.rt().get(x) * u64::from(k) * 10
            });
            Box::new(move |memo: &Memo<C, MemoryBackend>| {
                memo.runtime().get(memo.get_or_insert(scaled, 2))
            })
        };
        let memo = builder.open(&store, backend.clone()).unwrap();
        (store, memo, other, read)
    };
    {
        let (_store, memo, other, scaled) = open(false);
        assert_eq!(scaled(&memo), 60);
        assert_eq!(memo.runtime().get(memo.get_or_insert(other, 2)), 5);
        assert_eq!(memo.checkpoint().unwrap().written, 2);
    }
    let (_store, memo, other, scaled) = open(true);
    assert_eq!(scaled(&memo), 6000);
    assert_eq!(memo.runtime().get(memo.get_or_insert(other, 2)), 5);
    assert_eq!(runs.load(SeqCst), 3);
    assert_eq!(memo.stats().reused, 1);
}
