use std::sync::{Arc, Mutex, MutexGuard};

use crate::{Backend, Error, Position, ReadOutcome, Record, Result};

/// A log held in process memory.
///
/// Cloning yields another writable handle on the same log, so several
/// Stores in one process can share it as concurrent writers;
/// [`MemoryBackend::read_only`] yields a handle that can only read. The log
/// lives as long as any handle does. `sync` is a no-op: the log is as
/// durable as the process.
#[derive(Clone, Debug)]
pub struct MemoryBackend {
    log: Arc<Mutex<Log>>,
    writable: bool,
}

#[derive(Debug, Default)]
struct Log {
    tail: Position,
    snapshot: Option<(Position, Vec<Record>)>,
    records: Vec<(Position, Record)>,
}

impl MemoryBackend {
    /// Creates an empty log and a writable handle on it.
    pub fn new() -> Self {
        MemoryBackend {
            log: Arc::default(),
            writable: true,
        }
    }

    /// Returns a read-only handle on the same log.
    pub fn read_only(&self) -> Self {
        MemoryBackend {
            log: Arc::clone(&self.log),
            writable: false,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Log> {
        // Recovering from poison is sound because no method can panic
        // between its first and last mutation of the log, so a panicking
        // holder never leaves it half-updated.
        self.log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn require_writable(&self) -> Result<()> {
        if self.writable {
            Ok(())
        } else {
            Err(Error::ReadOnly)
        }
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for MemoryBackend {
    fn append(&self, records: &[Record]) -> Result<Position> {
        self.require_writable()?;
        let owned = records.to_vec();
        let mut log = self.lock();
        let first = log.tail.0 + 1;
        log.records.extend((first..).map(Position).zip(owned));
        log.tail = Position(log.tail.0 + records.len() as u64);
        Ok(log.tail)
    }

    fn sync(&self, _upto: Position) -> Result<()> {
        Ok(())
    }

    fn read_from(&self, after: Position, out: &mut Vec<(Position, Record)>) -> Result<ReadOutcome> {
        let log = self.lock();
        if let Some((snapshot_at, _)) = log.snapshot {
            if after < snapshot_at {
                return Ok(ReadOutcome::Compacted { snapshot_at });
            }
        }
        let start = log.records.partition_point(|(at, _)| *at <= after);
        out.extend_from_slice(&log.records[start..]);
        Ok(ReadOutcome::Records)
    }

    fn load_snapshot(&self) -> Result<Option<(Position, Vec<Record>)>> {
        Ok(self.lock().snapshot.clone())
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> Result<()> {
        self.require_writable()?;
        let mut log = self.lock();
        let snapshot_at = log.snapshot.as_ref().map_or(Position::ZERO, |(at, _)| *at);
        if upto < snapshot_at || upto > log.tail {
            return Err(Error::InvalidCompaction {
                upto,
                snapshot_at,
                tail: log.tail,
            });
        }
        let keep_from = log.records.partition_point(|(at, _)| *at <= upto);
        log.records.drain(..keep_from);
        log.snapshot = Some((upto, folded));
        Ok(())
    }

    fn writable(&self) -> bool {
        self.writable
    }
}
