#![cfg(feature = "sqlite")]

use std::path::PathBuf;

use incr_persist::conformance::Harness;
use incr_persist::SqliteBackend;
use tempfile::TempDir;

struct SqliteHarness;

struct Log {
    _dir: TempDir,
    path: PathBuf,
}

impl Harness for SqliteHarness {
    type Backend = SqliteBackend;
    type Log = Log;

    fn new_log(&mut self) -> Log {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("log.sqlite");
        Log { _dir: dir, path }
    }

    fn open(&mut self, log: &Log) -> SqliteBackend {
        SqliteBackend::open(&log.path).expect("open sqlite log")
    }

    fn open_read_only(&mut self, log: &Log) -> Option<SqliteBackend> {
        Some(SqliteBackend::open_read_only(&log.path).expect("open sqlite log read-only"))
    }

    fn multi_writer(&self) -> bool {
        true
    }
}

incr_persist::backend_conformance!(SqliteHarness);
