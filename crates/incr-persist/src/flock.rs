use std::fs::File;

/// A locked file that unlocks on drop. Closing alone releases an flock
/// only once every duplicate of its descriptor is closed, and a child that
/// another thread spawns holds one until it execs.
#[derive(Debug)]
pub(crate) struct Flock(pub(crate) File);

impl Drop for Flock {
    fn drop(&mut self) {
        // On failure, closing the file still releases the lock unless a
        // duplicate is open, and there is nowhere to report it.
        let _unlocked = self.0.unlock();
    }
}
