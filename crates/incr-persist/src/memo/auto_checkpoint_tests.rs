//! The automatic-checkpoint thread around staged batches and unlogged
//! scopes on `Shared`: a checkpoint skipped while one is open runs once it
//! closes, with no further compute, and a close after no skip wakes
//! nothing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::Memo;
use crate::memo::{BuildId, MemoBuilder, MemoOptions, MemoTable, MemoryMemo};
use crate::{Input, MemoryBackend, Options, Shared, Store};

const BUILD: u128 = 0x5c1b;
const TIMEOUT: Duration = Duration::from_secs(10);

struct World {
    store: Arc<Store<Shared, MemoryBackend>>,
    memo: Memo<Shared, MemoryBackend>,
    x: Input<u64>,
    t: MemoTable<u32, u64>,
}

fn open(trigger: u64) -> World {
    let store = Arc::new(Store::open(MemoryBackend::new(), Options::default()).unwrap());
    let x = store.input("x", 1_u64).unwrap();
    let xh = x.handle();
    let mut builder =
        MemoBuilder::new(MemoOptions::new(BuildId::Fixed(BUILD)).checkpoint_after_entries(trigger));
    let t = builder.table::<u32, u64>("t", 1);
    builder.define(t, move |ctx, &k| ctx.rt().get(xh) + u64::from(k));
    let memo = builder.open(&store, MemoryMemo::new()).unwrap();
    World { store, memo, x, t }
}

impl World {
    fn read(&self, k: u32) -> u64 {
        let q = self.memo.get_or_insert(self.t, k);
        self.memo.runtime().get(q)
    }

    /// Reads keys `0..trigger` and waits until the woken thread skipped
    /// its checkpoint and went back to waiting.
    fn skip_one(&self, trigger: u32) {
        for k in 0..trigger {
            self.read(k);
        }
        assert!(
            self.memo.core.wake.wait_for_held_skips(1, TIMEOUT),
            "the thread never skipped a checkpoint"
        );
    }

    fn wait_for_checkpoints(&self, n: u64) {
        let deadline = Instant::now() + TIMEOUT;
        while self.memo.stats().checkpoints < n {
            assert!(Instant::now() < deadline, "no automatic checkpoint ran");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn assert_saved_without_errors(&self, entries: u64) {
        let stats = self.memo.stats();
        assert_eq!(
            (
                stats.entries,
                stats.unsaved_entries,
                stats.auto_checkpoint_errors
            ),
            (entries, 0, 0)
        );
    }
}

#[test]
fn a_checkpoint_skipped_for_a_staged_batch_runs_when_it_commits() {
    let w = open(2);
    let mut batch = w.store.staged_batch().unwrap();
    batch.set(w.x, 5).unwrap();
    w.skip_one(2);
    batch.commit().unwrap();
    w.wait_for_checkpoints(1);
    w.assert_saved_without_errors(2);
}

#[test]
fn a_checkpoint_skipped_for_an_unlogged_scope_runs_when_it_closes() {
    let w = open(2);
    let scope = w.store.unlogged_scope().unwrap();
    w.skip_one(2);
    scope.close().unwrap();
    w.wait_for_checkpoints(1);
    w.assert_saved_without_errors(2);
}

#[test]
fn closing_a_batch_or_scope_after_no_skip_wakes_nothing() {
    let w = open(3);
    let dues = w.memo.core.wake.dues();
    let mut batch = w.store.staged_batch().unwrap();
    batch.set(w.x, 5).unwrap();
    w.read(0);
    batch.commit().unwrap();
    let scope = w.store.unlogged_scope().unwrap();
    w.read(1);
    scope.close().unwrap();
    assert_eq!(w.memo.core.wake.dues(), dues);
    assert_eq!(w.memo.stats().checkpoints, 0);
}

/// The skip a close resumed is spent: the next batch's close, with the
/// trigger not reached, wakes nothing.
#[test]
fn a_close_resumes_a_skip_once() {
    let w = open(2);
    let mut batch = w.store.staged_batch().unwrap();
    w.skip_one(2);
    batch.commit().unwrap();
    w.wait_for_checkpoints(1);
    let dues = w.memo.core.wake.dues();
    let mut batch = w.store.staged_batch().unwrap();
    batch.set(w.x, 5).unwrap();
    w.read(7);
    batch.commit().unwrap();
    assert_eq!(w.memo.core.wake.dues(), dues);
    assert_eq!(w.memo.stats().checkpoints, 1);
}
