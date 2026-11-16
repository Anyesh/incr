//! Resident memory across a 1% delta checkpoint of a 500000-entry memo
//! restored from a file memo. Ignored by default: run with
//! `cargo test --release -p incr-persist --test memo_rss -- --ignored`.
#![cfg(all(target_os = "linux", target_env = "gnu"))]

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use incr_persist::memo::{BuildId, FileMemo, Memo, MemoBuilder, MemoCells, MemoOptions, MemoTable};
use incr_persist::{Input, Local, MemoryBackend, Options, Shared, Store};

const BUILD: u128 = 0x255;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// VmRSS is per process, so the two tests must not overlap.
fn exclusive() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}
const ENTRIES: u32 = 500_000;
const DIRTY: u32 = ENTRIES / 100;
const VALUE: usize = 100;
const SLACK_PERCENT: u64 = 5;

fn vm_rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmRSS:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

struct Rig<C: MemoCells> {
    store: Arc<Store<C, MemoryBackend>>,
    memo: Memo<C, MemoryBackend>,
    epoch: Input<u64>,
    table: MemoTable<u32, Vec<u8>>,
}

impl<C: MemoCells> Rig<C> {
    fn open(log: &MemoryBackend, dir: &std::path::Path, options: MemoOptions) -> Self {
        let store = Arc::new(Store::open(log.clone(), Options::default()).unwrap());
        let epoch = store.input("epoch", 0_u64).unwrap();
        let handle = epoch.handle();
        let mut builder = MemoBuilder::new(options);
        let table = builder.table::<u32, Vec<u8>>("blob", 1);
        builder.define(table, move |ctx, &k| {
            let seed = if k < DIRTY { ctx.rt().get(handle) } else { 0 };
            vec![(u64::from(k) ^ seed) as u8; VALUE]
        });
        let memo = builder.open(&store, FileMemo::open(dir).unwrap()).unwrap();
        Rig {
            store,
            memo,
            epoch,
            table,
        }
    }

    fn read(&self, k: u32) -> usize {
        let q = self.memo.get_or_insert(self.table, k);
        self.memo.runtime().get(q).len()
    }

    fn read_all(&self) {
        for k in 0..ENTRIES {
            assert_eq!(self.read(k), VALUE);
        }
    }

    fn dirty_one_percent(&self) {
        self.store.set(self.epoch, 1).unwrap();
        for k in 0..DIRTY {
            self.read(k);
        }
    }
}

fn options() -> MemoOptions {
    MemoOptions::new(BuildId::Fixed(BUILD)).max_bytes(u64::MAX)
}

fn written_memo() -> (tempfile::TempDir, MemoryBackend) {
    let dir = tempfile::tempdir().unwrap();
    let log = MemoryBackend::new();
    let rig = Rig::<Local>::open(&log, dir.path(), options());
    rig.read_all();
    assert!(!rig.memo.checkpoint().unwrap().delta);
    (dir, log)
}

fn assert_within_slack(before: u64, after: u64) {
    let limit = before + before * SLACK_PERCENT / 100;
    assert!(
        after <= limit,
        "VmRSS {before} KiB before the checkpoint, {after} KiB after: over {SLACK_PERCENT}%"
    );
}

#[test]
#[ignore = "500000 entries; run in release"]
fn a_one_percent_checkpoint_keeps_resident_memory_flat_on_local() {
    let _exclusive = exclusive();
    let (dir, log) = written_memo();
    let rig = Rig::<Local>::open(&log, dir.path(), options());
    rig.read_all();
    let before = vm_rss_kib();
    rig.dirty_one_percent();
    let done = rig.memo.checkpoint().unwrap();
    assert!(done.delta, "{done:?}");
    assert_eq!(done.fresh_written, DIRTY as usize);
    assert_within_slack(before, vm_rss_kib());
}

#[test]
#[ignore = "500000 entries; run in release"]
fn a_one_percent_automatic_checkpoint_keeps_resident_memory_flat_on_shared() {
    let _exclusive = exclusive();
    let (dir, log) = written_memo();
    let rig = Rig::<Shared>::open(
        &log,
        dir.path(),
        options().checkpoint_after_entries(u64::from(DIRTY)),
    );
    rig.read_all();
    assert_eq!(rig.memo.stats().checkpoints, 0);
    let before = vm_rss_kib();
    rig.dirty_one_percent();
    let deadline = Instant::now() + Duration::from_secs(120);
    while rig.memo.stats().delta_checkpoints < 1 {
        assert!(
            Instant::now() < deadline,
            "no automatic delta checkpoint ran"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_within_slack(before, vm_rss_kib());
}
