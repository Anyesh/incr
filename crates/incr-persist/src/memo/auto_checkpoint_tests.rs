//! The automatic-checkpoint thread around staged batches and unlogged
//! scopes on `Shared`: a checkpoint skipped while one is open runs once it
//! closes, with no further compute, and a close after no skip wakes
//! nothing.

use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use super::Memo;
use crate::memo::table_core::COMPUTED_PARK;
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

    fn wait_for_entries(&self, n: u64) {
        let deadline = Instant::now() + TIMEOUT;
        while self.memo.stats().entries < n {
            assert!(
                Instant::now() < deadline,
                "the entry was not saved after it landed"
            );
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

/// Parks `w.read(k)` on a thread after its compute recorded the entry
/// dirty and before the node lands; the returned sender releases it.
fn park_read(w: &Arc<World>, k: u32) -> (mpsc::Sender<()>, std::thread::JoinHandle<u64>) {
    let (parked_tx, parked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let reader = {
        let w = Arc::clone(w);
        std::thread::spawn(move || {
            let release_rx = Mutex::new(release_rx);
            COMPUTED_PARK.set(Some(Box::new(move || {
                parked_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
            })));
            w.read(k)
        })
    };
    parked_rx.recv_timeout(TIMEOUT).unwrap();
    (release_tx, reader)
}

/// The trigger wakes the thread from inside the compute that reached it,
/// before the engine lands the node, so the woken checkpoint cannot save
/// that entry. The thread must checkpoint again once the node lands, with
/// no further compute.
#[test]
fn an_entry_that_lands_after_its_woken_checkpoint_is_saved() {
    let w = Arc::new(open(1));
    assert!(w.memo.core.wake.wait_until_blocked(TIMEOUT));
    let (release, reader) = park_read(&w, 0);
    assert!(
        w.memo.core.wake.wait_until_blocked(TIMEOUT),
        "the woken thread never went back to waiting"
    );
    release.send(()).unwrap();
    assert_eq!(reader.join().unwrap(), 1);
    w.wait_for_entries(1);
    w.assert_saved_without_errors(1);
}

/// A carried entry that lands with fewer than the trigger's entries
/// unsaved stays unsaved: its landing alone runs no checkpoint.
#[test]
fn a_carried_entry_that_lands_below_the_trigger_waits_for_more_computes() {
    let w = Arc::new(open(3));
    w.read(0);
    w.read(1);
    let (release, reader) = park_read(&w, 2);
    assert!(w.memo.core.wake.wait_until_blocked(TIMEOUT));
    w.wait_for_checkpoints(1);
    release.send(()).unwrap();
    assert_eq!(reader.join().unwrap(), 3);
    assert!(
        w.memo.core.wake.wait_for_landings(1, TIMEOUT),
        "the thread never saw the carried entry land"
    );
    assert!(w.memo.core.wake.wait_until_blocked(TIMEOUT));
    let stats = w.memo.stats();
    assert_eq!(
        (stats.checkpoints, stats.entries, stats.unsaved_entries),
        (1, 2, 1)
    );
    w.read(3);
    w.read(4);
    w.wait_for_checkpoints(2);
}

/// A second checkpoint before a carried entry lands carries it without
/// counting it again, so its landing must count it for the trigger.
#[test]
fn an_entry_carried_twice_is_saved_when_it_lands() {
    let w = Arc::new(open(1));
    assert!(w.memo.core.wake.wait_until_blocked(TIMEOUT));
    let (release, reader) = park_read(&w, 0);
    assert!(w.memo.core.wake.wait_until_blocked(TIMEOUT));
    assert!(w.memo.checkpoint().unwrap().unchanged);
    assert!(w.memo.core.wake.wait_until_blocked(TIMEOUT));
    assert_eq!(w.memo.stats().unsaved_entries, 0);
    release.send(()).unwrap();
    assert_eq!(reader.join().unwrap(), 1);
    w.wait_for_entries(1);
    w.assert_saved_without_errors(1);
}
