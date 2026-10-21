use std::path::PathBuf;

use incr_persist::memo::MemoCells;
use incr_persist::{Local, Shared};
use tempfile::TempDir;

/// The runtimes a bench runs on, each under its own group names.
pub trait BenchCells: MemoCells {
    /// Appended to a group's name; empty for Local, so its groups keep the
    /// names their recorded results were taken under.
    const SUFFIX: &'static str;
}

impl BenchCells for Local {
    const SUFFIX: &'static str = "";
}

impl BenchCells for Shared {
    const SUFFIX: &'static str = "_shared";
}

/// A fresh directory under the workspace's `target/` (or
/// `INCR_PERSIST_BENCH_DIR`), never under a tmpfs `/tmp`, where fsync
/// would cost nothing.
pub fn bench_dir() -> TempDir {
    let base = std::env::var_os("INCR_PERSIST_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/incr-persist-bench")
        });
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(base).unwrap()
}
