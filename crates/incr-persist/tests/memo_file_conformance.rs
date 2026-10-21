use std::path::PathBuf;

use incr_persist::memo::{conformance::MemoHarness, FileMemo};
use tempfile::TempDir;

#[derive(Default)]
struct FileMemoHarness {
    dirs: Vec<TempDir>,
}

impl MemoHarness for FileMemoHarness {
    type Backend = FileMemo;
    type Store = PathBuf;

    fn new_store(&mut self) -> PathBuf {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memo");
        self.dirs.push(dir);
        path
    }

    fn open(&mut self, store: &PathBuf) -> FileMemo {
        FileMemo::open(store).unwrap()
    }

    fn exclusive_lock(&self) -> bool {
        true
    }
}

incr_persist::memo_backend_conformance!(FileMemoHarness::default());
