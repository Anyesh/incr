use std::path::PathBuf;

use tempfile::TempDir;

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
