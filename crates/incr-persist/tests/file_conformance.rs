use std::path::PathBuf;

use incr_persist::conformance::Harness;
use incr_persist::FileBackend;
use tempfile::TempDir;

#[derive(Default)]
struct FileHarness {
    dirs: Vec<TempDir>,
}

impl Harness for FileHarness {
    type Backend = FileBackend;
    type Log = PathBuf;

    fn new_log(&mut self) -> PathBuf {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log");
        self.dirs.push(dir);
        path
    }

    fn open(&mut self, log: &PathBuf) -> FileBackend {
        FileBackend::open(log).unwrap()
    }

    fn open_read_only(&mut self, log: &PathBuf) -> Option<FileBackend> {
        Some(FileBackend::open_read_only(log).unwrap())
    }

    fn multi_writer(&self) -> bool {
        false
    }
}

incr_persist::backend_conformance!(FileHarness::default());
