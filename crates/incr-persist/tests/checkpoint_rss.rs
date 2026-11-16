//! Peak resident memory of a store checkpoint over 600000 rows of about
//! 100 bytes on the Local runtime. Ignored by default: run each test alone,
//! `cargo test --release -p incr-persist --features sqlite --test
//! checkpoint_rss -- --ignored --test-threads 1 --nocapture`.

#![cfg(all(target_os = "linux", target_env = "gnu"))]

use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use incr_persist::{Backend, FileBackend, Local, Options, PendingWrite, PersistedKeyed, Store};

const ROWS: u32 = 600_000;
const BATCH: u32 = 10_000;
const SLACK_PERCENT: f64 = 5.0;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// Resident memory is per process, so the tests must not overlap.
fn exclusive() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

fn status_kib(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with(field)).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

/// A row of 96 bytes, which with its key encodes to about 100.
fn row(k: u32) -> String {
    format!("{k:010}{}", "x".repeat(86))
}

/// Checkpoints `store` and returns the VmHWM rise during the checkpoint
/// over the VmRSS before it, in percent of the latter.
fn checkpoint_rise<B: Backend>(name: &str, store: &Store<Local, B>) -> f64 {
    let before = status_kib("VmRSS:");
    // Resets VmHWM to the current VmRSS.
    std::fs::write("/proc/self/clear_refs", "5").unwrap();
    let started = Instant::now();
    store.checkpoint().unwrap();
    let elapsed = started.elapsed();
    let peak = status_kib("VmHWM:");
    let rise = peak.saturating_sub(before);
    let percent = rise as f64 * 100.0 / before as f64;
    eprintln!(
        "{name}: VmRSS {before} KiB before, VmHWM {peak} KiB during, rise {rise} KiB \
         ({percent:.2}%), checkpoint {elapsed:.2?}"
    );
    percent
}

fn keyed_rise<B: Backend>(name: &str, backend: B) -> f64 {
    let store = Store::<Local, _>::open(backend, Options::default()).unwrap();
    let rows: PersistedKeyed<u32, String, Local> = store.keyed_collection("rows").unwrap();
    for start in (0..ROWS).step_by(BATCH as usize) {
        let mut pending = PendingWrite::new();
        for k in start..start + BATCH {
            pending.upsert(&rows, k, row(k));
        }
        store.write_pending(pending).unwrap();
    }
    checkpoint_rise(name, &store)
}

#[test]
#[ignore = "600000 rows; run in release, one test at a time"]
fn a_keyed_checkpoint_on_file_keeps_peak_memory_within_slack() {
    let _one = exclusive();
    let dir = tempfile::tempdir().unwrap();
    let percent = keyed_rise("keyed, file", FileBackend::open(dir.path()).unwrap());
    assert!(
        percent <= SLACK_PERCENT,
        "{percent:.2}% over {SLACK_PERCENT}%"
    );
}

#[cfg(feature = "sqlite")]
#[test]
#[ignore = "600000 rows; run in release, one test at a time"]
fn a_keyed_checkpoint_on_sqlite_keeps_peak_memory_within_slack() {
    let _one = exclusive();
    let dir = tempfile::tempdir().unwrap();
    let backend = incr_persist::SqliteBackend::open(dir.path().join("log.db")).unwrap();
    let percent = keyed_rise("keyed, sqlite", backend);
    assert!(
        percent <= SLACK_PERCENT,
        "{percent:.2}% over {SLACK_PERCENT}%"
    );
}

/// Reports the figure only: a plain collection's fold keeps 16 bytes per
/// element to order them.
#[test]
#[ignore = "600000 rows; run in release, one test at a time"]
fn a_plain_collection_checkpoint_on_file_reports_its_peak_memory() {
    let _one = exclusive();
    let dir = tempfile::tempdir().unwrap();
    let store = Store::<Local, _>::open(FileBackend::open(dir.path()).unwrap(), Options::default())
        .unwrap();
    let values = store.collection::<String>("values").unwrap();
    for start in (0..ROWS).step_by(BATCH as usize) {
        store
            .batch(|b| {
                for k in start..start + BATCH {
                    b.insert(&values, row(k));
                }
            })
            .unwrap();
    }
    checkpoint_rise("plain, file", &store);
}
