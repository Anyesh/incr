#![cfg(feature = "sqlite")]

use std::path::PathBuf;

use incr_persist::memo::{conformance::MemoHarness, SqliteMemo};
use tempfile::TempDir;

struct SqliteHarness;

struct Db {
    _dir: TempDir,
    path: PathBuf,
}

impl MemoHarness for SqliteHarness {
    type Backend = SqliteMemo;
    type Store = Db;

    fn new_store(&mut self) -> Db {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("memo.sqlite");
        Db { _dir: dir, path }
    }

    fn open(&mut self, store: &Db) -> SqliteMemo {
        SqliteMemo::open(&store.path).expect("open sqlite memo")
    }

    fn exclusive_lock(&self) -> bool {
        true
    }
}

incr_persist::memo_backend_conformance!(SqliteHarness);
