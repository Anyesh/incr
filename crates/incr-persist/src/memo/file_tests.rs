use super::*;
use crate::memo::{
    decode_entry, write_checkpoint, CheckpointInput, Entry, StoredDep, TableDesc, DEFAULT_MAX_BYTES,
};

fn header(seq: u64) -> GenHeader {
    GenHeader {
        build: 9,
        seq,
        tables: vec![desc()],
    }
}

fn desc() -> TableDesc {
    TableDesc {
        name: "t".into(),
        version: 1,
        key_type: "u8".into(),
        value_type: "u8".into(),
    }
}

fn entry(key: u8) -> Entry {
    Entry {
        table: 0,
        key: vec![key],
        value: vec![key; 5],
        deps: vec![StoredDep::External {
            name: format!("x{key}"),
            fp: u128::from(key),
        }],
        last_used: 0,
    }
}

fn entries(keys: std::ops::Range<u8>) -> Vec<EncodedEntry> {
    keys.map(|key| EncodedEntry::new(&entry(key)).unwrap())
        .collect()
}

fn store() -> (tempfile::TempDir, FileMemo) {
    let dir = tempfile::tempdir().unwrap();
    let memo = FileMemo::open(dir.path()).unwrap();
    (dir, memo)
}

fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

fn gen_file(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("{GEN_PREFIX}{id}"))
}

/// Loads through a fresh handle, so nothing is served from a cached file.
fn reload(dir: &Path) -> Option<Generation> {
    FileMemo::open(dir).unwrap().load().unwrap()
}

fn assert_whole(generation: &Generation, written: &Generation) {
    assert_eq!(generation.id(), written.id());
    assert_eq!(generation.header(), written.header());
    assert_eq!(generation.items(), written.items());
}

fn assert_empty(generation: Option<Generation>, id: u64) {
    let generation = generation.expect("a damaged generation still loads");
    assert_eq!(generation.id(), GenId(id));
    assert!(generation.items().is_empty());
    assert_eq!(generation.header(), Generation::empty(GenId(id)).header());
}

#[test]
fn every_byte_flip_outside_entries_loads_empty_and_inside_fails_decode() {
    let (dir, memo) = store();
    let written = memo
        .write_generation(None, GenId(1), &header(1), &entries(0..3))
        .unwrap();
    let path = gen_file(dir.path(), 1);
    let pristine = fs::read(&path).unwrap();
    let entries_at = written.items().iter().map(|i| i.loc).min().unwrap();
    let entries_end = written
        .items()
        .iter()
        .map(|i| i.loc + u64::from(i.len))
        .max()
        .unwrap();
    for at in 0..pristine.len() {
        let mut damaged = pristine.clone();
        damaged[at] ^= 0x10;
        fs::write(&path, &damaged).unwrap();
        let loaded = reload(dir.path());
        let at = at as u64;
        if (entries_at..entries_end).contains(&at) {
            let loaded = loaded.unwrap();
            assert_whole(&loaded, &written);
            let handle = FileMemo::open(dir.path()).unwrap();
            let item = loaded
                .items()
                .iter()
                .find(|i| (i.loc..i.loc + u64::from(i.len)).contains(&at))
                .unwrap();
            let frame = handle.read_entry(GenId(1), item).unwrap().unwrap();
            assert!(decode_entry(&frame).is_err(), "flip at {at} decoded");
        } else {
            assert_empty(loaded, 1);
        }
    }
}

#[test]
fn every_truncation_loads_empty() {
    let (dir, memo) = store();
    memo.write_generation(None, GenId(1), &header(1), &entries(0..3))
        .unwrap();
    let path = gen_file(dir.path(), 1);
    let pristine = fs::read(&path).unwrap();
    for len in 0..pristine.len() {
        fs::write(&path, &pristine[..len]).unwrap();
        assert_empty(reload(dir.path()), 1);
    }
}

/// A handle that opened the generation before it was truncated reads the
/// short frame; a handle opening it afterwards finds no valid footer and
/// treats the generation as gone.
#[test]
fn a_truncated_generation_reads_short_or_not_at_all() {
    let (dir, memo) = store();
    let written = memo
        .write_generation(None, GenId(1), &header(1), &entries(0..3))
        .unwrap();
    let last = *written.items().iter().max_by_key(|i| i.loc).unwrap();
    let reader = FileMemo::open(dir.path()).unwrap();
    assert_whole(&reader.load().unwrap().unwrap(), &written);
    let file = OpenOptions::new()
        .write(true)
        .open(gen_file(dir.path(), 1))
        .unwrap();
    file.set_len(last.loc + 3).unwrap();
    let frame = reader.read_entry(GenId(1), &last).unwrap().unwrap();
    assert_eq!(frame.len(), 3);
    assert!(decode_entry(&frame).is_err());
    let fresh = FileMemo::open(dir.path()).unwrap();
    assert_eq!(fresh.read_entry(GenId(1), &last).unwrap(), None);
}

#[test]
fn swapped_generation_files_are_not_served() {
    let (dir, memo) = store();
    let first = memo
        .write_generation(None, GenId(1), &header(1), &entries(0..2))
        .unwrap();
    memo.write_generation(Some(GenId(1)), GenId(2), &header(2), &entries(0..4))
        .unwrap();
    let (one, two) = (gen_file(dir.path(), 1), gen_file(dir.path(), 2));
    let swap = dir.path().join("swap");
    fs::rename(&one, &swap).unwrap();
    fs::rename(&two, &one).unwrap();
    fs::rename(&swap, &two).unwrap();

    assert_empty(reload(dir.path()), 2);
    let handle = FileMemo::open(dir.path()).unwrap();
    for item in first.items() {
        assert_eq!(handle.read_entry(GenId(1), item).unwrap(), None);
    }
}

#[test]
fn a_vanished_generation_file_is_a_miss_and_the_next_write_replaces_it() {
    let (dir, memo) = store();
    let written = memo
        .write_generation(None, GenId(1), &header(1), &entries(0..2))
        .unwrap();
    fs::remove_file(gen_file(dir.path(), 1)).unwrap();
    let handle = FileMemo::open(dir.path()).unwrap();
    assert_empty(handle.load().unwrap(), 1);
    assert_eq!(
        handle.read_entry(GenId(1), &written.items()[0]).unwrap(),
        None
    );
    handle
        .write_generation(Some(GenId(1)), GenId(2), &header(2), &entries(0..1))
        .unwrap();
    assert_eq!(reload(dir.path()).unwrap().items().len(), 1);
}

#[test]
fn a_damaged_current_names_the_highest_generation() {
    let (dir, memo) = store();
    memo.write_generation(None, GenId(1), &header(1), &entries(0..2))
        .unwrap();
    let second = memo
        .write_generation(Some(GenId(1)), GenId(2), &header(2), &entries(0..3))
        .unwrap();
    let current = dir.path().join(CURRENT);
    let mut bytes = fs::read(&current).unwrap();
    bytes[0] ^= 1;
    fs::write(&current, &bytes).unwrap();
    assert_whole(&reload(dir.path()).unwrap(), &second);

    fs::write(&current, b"garbage").unwrap();
    fs::remove_file(gen_file(dir.path(), 1)).unwrap();
    fs::remove_file(gen_file(dir.path(), 2)).unwrap();
    assert_empty(reload(dir.path()), 0);
    memo.write_generation(Some(GenId(0)), GenId(1), &header(1), &entries(0..1))
        .unwrap();
    assert_eq!(reload(dir.path()).unwrap().id(), GenId(1));
}

/// The states a crash leaves at each step of a write: a temp file only, or
/// the new generation renamed into place with `MEMO_CURRENT` not yet
/// replaced. Both load the base whole, and the next checkpoint lock clears
/// them.
#[test]
fn an_interrupted_write_leaves_the_base_current_and_is_cleared_by_the_next_lock() {
    let (dir, memo) = store();
    let base = memo
        .write_generation(None, GenId(1), &header(1), &entries(0..2))
        .unwrap();
    let (other_dir, other) = store();
    other
        .write_generation(None, GenId(2), &header(2), &entries(0..3))
        .unwrap();
    fs::copy(gen_file(other_dir.path(), 2), gen_file(dir.path(), 2)).unwrap();
    fs::write(dir.path().join(format!("{TMP_PREFIX}3.1.0")), b"torn").unwrap();
    fs::write(dir.path().join(CURRENT_TMP), b"torn").unwrap();

    assert_whole(&reload(dir.path()).unwrap(), &base);
    let guard = memo.lock_checkpoint().unwrap();
    assert_eq!(names(dir.path()), [CURRENT, "MEMO_GEN.1", LOCK, PUBLISH]);
    let next = memo
        .write_generation(Some(GenId(1)), GenId(2), &header(2), &entries(0..1))
        .unwrap();
    drop(guard);
    assert_whole(&reload(dir.path()).unwrap(), &next);
}

#[test]
fn a_failed_write_leaves_no_temp_file() {
    let (dir, memo) = store();
    memo.write_generation(None, GenId(1), &header(1), &entries(0..2))
        .unwrap();
    let stale = memo.write_generation(None, GenId(2), &header(2), &entries(0..2));
    assert!(matches!(stale, Err(Error::Busy)));
    assert!(names(dir.path())
        .iter()
        .all(|name| !name.starts_with(TMP_PREFIX)));
}

#[test]
fn a_held_lock_makes_a_checkpoint_busy_and_writes_nothing() {
    let (dir, memo) = store();
    let other = FileMemo::open(dir.path()).unwrap();
    let guard = other.lock_checkpoint().unwrap();
    let input = || CheckpointInput {
        build: 9,
        tables: vec![desc()],
        fresh: vec![entry(1)],
        max_bytes: DEFAULT_MAX_BYTES,
    };
    assert!(matches!(write_checkpoint(&memo, input()), Err(Error::Busy)));
    assert!(memo.load().unwrap().is_none());
    drop(guard);
    let done = write_checkpoint(&memo, input()).unwrap();
    assert_eq!(done.stats.written, 1);
}

#[test]
fn checkpoints_merge_across_handles_and_reopen() {
    let (dir, memo) = store();
    let checkpoint = |memo: &FileMemo, keys: std::ops::Range<u8>| {
        let done = write_checkpoint(
            memo,
            CheckpointInput {
                build: 9,
                tables: vec![desc()],
                fresh: keys.map(entry).collect(),
                max_bytes: DEFAULT_MAX_BYTES,
            },
        )
        .unwrap();
        if let Some(old) = done.replaced {
            memo.remove_generation(old).unwrap();
        }
        done.generation
    };
    checkpoint(&memo, 0..3);
    let other = FileMemo::open(dir.path()).unwrap();
    let last = checkpoint(&other, 3..5);
    assert_eq!(names(dir.path()), [CURRENT, "MEMO_GEN.2", LOCK, PUBLISH]);

    let reopened = FileMemo::open(dir.path()).unwrap();
    let loaded = reopened.load().unwrap().unwrap();
    assert_whole(&loaded, &last);
    let mut keys: Vec<_> = loaded
        .items()
        .iter()
        .map(|item| {
            let frame = reopened.read_entry(loaded.id(), item).unwrap().unwrap();
            decode_entry(&frame).unwrap().key[0]
        })
        .collect();
    keys.sort();
    assert_eq!(keys, [0, 1, 2, 3, 4]);
}

#[test]
fn a_log_in_the_same_directory_leaves_the_memo_alone() {
    let (dir, memo) = store();
    let first = memo
        .write_generation(None, GenId(1), &header(1), &entries(0..2))
        .unwrap();
    let log = crate::FileBackend::open(dir.path()).unwrap();
    let record = crate::Record::Delete { key: "k".into() };
    let at = crate::Backend::append(&log, &[record]).unwrap();
    crate::Backend::compact(&log, at, Vec::new()).unwrap();
    drop(log);
    assert_whole(&reload(dir.path()).unwrap(), &first);
}
