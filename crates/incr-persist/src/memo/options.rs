use crate::memo::DEFAULT_MAX_BYTES;
use crate::Result;

/// What a memo's entries were computed by. Entries written under another
/// build id are dropped whole on open, so it must change whenever a memo
/// closure's code may have.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildId {
    /// The xxh3-128 of `/proc/self/exe`, which names the running inode
    /// even after the file is replaced in place. Elsewhere `current_exe()`
    /// names a path an upgrade can repoint, so there is no such variant
    /// there.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    CurrentExe,
    /// A caller-computed id, for example from a `build.rs` that hashes the
    /// crate's sources and the rustc version into an environment variable:
    /// `BuildId::Fixed(memo::fingerprint(env!("MY_BUILD_ID").as_bytes()))`.
    Fixed(u128),
}

/// How a memo is opened. There is no `Default`: the build id is required.
#[derive(Clone, Debug)]
pub struct MemoOptions {
    pub(crate) build_id: BuildId,
    pub(crate) salt: Vec<u8>,
    pub(crate) max_bytes: u64,
    pub(crate) auto_checkpoint_secs: Option<u64>,
}

impl MemoOptions {
    pub fn new(build_id: BuildId) -> Self {
        MemoOptions {
            build_id,
            salt: Vec::new(),
            max_bytes: DEFAULT_MAX_BYTES,
            auto_checkpoint_secs: None,
        }
    }

    /// Bytes folded into the build id, for configuration the closures read
    /// that the build id doesn't cover.
    pub fn salt(mut self, salt: impl Into<Vec<u8>>) -> Self {
        self.salt = salt.into();
        self
    }

    /// The size a checkpoint evicts the oldest entries down to.
    pub fn max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Checkpoints on a background thread every `secs` seconds, joined
    /// when the memo drops. `Shared` only: opening a `Local` memo with it
    /// fails with `Error::AutoCheckpointNeedsShared`.
    pub fn auto_checkpoint_secs(mut self, secs: u64) -> Self {
        self.auto_checkpoint_secs = Some(secs);
        self
    }

    pub(crate) fn build(&self) -> Result<u128> {
        let id = match &self.build_id {
            #[cfg(any(target_os = "linux", target_os = "android"))]
            BuildId::CurrentExe => {
                hash_file("/proc/self/exe").map_err(crate::Error::BuildIdUnavailable)?
            }
            BuildId::Fixed(id) => *id,
        };
        let mut bytes = id.to_le_bytes().to_vec();
        bytes.extend_from_slice(&self.salt);
        Ok(xxhash_rust::xxh3::xxh3_128(&bytes))
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn hash_file(path: &str) -> std::io::Result<u128> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    let mut buf = vec![0; 1 << 16];
    loop {
        match file.read(&mut buf)? {
            0 => return Ok(hasher.digest128()),
            n => hasher.update(&buf[..n]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_fingerprint_folds_the_id_and_the_salt() {
        let build = |options: MemoOptions| options.build().unwrap();
        let fixed = MemoOptions::new(BuildId::Fixed(7));
        assert_eq!(build(fixed.clone()), build(fixed.clone()));
        assert_ne!(
            build(fixed.clone()),
            build(MemoOptions::new(BuildId::Fixed(8)))
        );
        assert_ne!(build(fixed.clone()), build(fixed.clone().salt("a")));
        assert_ne!(build(fixed.clone().salt("a")), build(fixed.salt("b")));
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn current_exe_hashes_the_running_binary() {
        let exe = MemoOptions::new(BuildId::CurrentExe);
        assert_eq!(exe.build().unwrap(), exe.build().unwrap());
        let bytes = std::fs::read("/proc/self/exe").unwrap();
        let fixed = MemoOptions::new(BuildId::Fixed(xxhash_rust::xxh3::xxh3_128(&bytes)));
        assert_eq!(exe.build().unwrap(), fixed.build().unwrap());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn an_unreadable_executable_is_an_error() {
        let err = hash_file("/proc/self/no such file").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
