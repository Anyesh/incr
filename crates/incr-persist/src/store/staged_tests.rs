use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use super::staged::{StagedPark, STAGED_PARK};
use super::{Input, Options, PendingWrite, PersistedCollection, PersistedKeyed, Store};
use crate::{Cells, Error, Local, MemoryBackend, Position, Record, Shared};

/// A park point that is never reached would otherwise hang the test.
const PARK_TIMEOUT: Duration = Duration::from_secs(10);

fn a_record_that_fails_to_apply_poisons<C: Cells>() {
    let store: Store<C, _> = Store::open(MemoryBackend::new(), Options::default()).unwrap();
    let x = store.input("x", 0_u32).unwrap();
    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 1).unwrap();
    let bad = Record::Set {
        key: "x".to_owned(),
        value: vec![0xff; 8],
    };
    let failed = {
        let mut state = store.state().unwrap();
        store.apply_staged(&mut state, bad, &mut Vec::new())
    };
    assert!(matches!(failed, Err(Error::TypeMismatch { .. })));
    assert!(matches!(staged.set(x, 2), Err(Error::Poisoned)));
    assert!(matches!(staged.commit(), Err(Error::Poisoned)));
}

#[test]
fn a_record_that_fails_to_apply_poisons_local() {
    a_record_that_fails_to_apply_poisons::<Local>();
}

#[test]
fn a_record_that_fails_to_apply_poisons_shared() {
    a_record_that_fails_to_apply_poisons::<Shared>();
}

/// One store's handles on the keys a staged batch writes.
struct Keys {
    store: Store<Shared, MemoryBackend>,
    x: Input<u32>,
    c: PersistedCollection<u32, Shared>,
    k: PersistedKeyed<u32, String, Shared>,
}

/// What a reader sees: x, the collection's values, rows 1 and 2.
type Seen = (u32, Vec<u32>, [Option<String>; 2]);

impl Keys {
    fn open(log: &MemoryBackend) -> Self {
        let store = Store::open(log.clone(), Options::default()).unwrap();
        let x = store.input("x", 0_u32).unwrap();
        let c = store.collection("c").unwrap();
        let k = store.keyed_collection("k").unwrap();
        Keys { store, x, c, k }
    }

    fn seen(&self) -> Seen {
        let rt = self.store.runtime();
        let mut values = self.c.handle().values(rt);
        values.sort_unstable();
        (
            rt.get(self.x.handle()),
            values,
            [self.k.row(&1), self.k.row(&2)],
        )
    }

    /// Catches up with the log; its position, then what it then reads.
    fn refreshed(&self) -> (Position, Seen) {
        (self.store.refresh().unwrap(), self.seen())
    }
}

fn progress(seen: &Seen) -> (u32, usize, usize) {
    let (x, values, rows) = seen;
    (*x, values.len(), rows.iter().flatten().count())
}

/// A Shared staged batch parks after each applied record and between
/// its run landing and its position moving, while a reader samples the
/// writing store's runtime and a second store over the same log
/// refreshes. The writer's values never go backwards; the second store
/// sees the old position with the old values or the run's end with all
/// of them, nothing between; a `stats` call on the writing store waits
/// for the commit and then reads the run's end; and a reopen after the
/// commit reads what the writer holds live.
#[test]
fn a_shared_reader_never_sees_a_staged_position_without_its_values() {
    let log = MemoryBackend::new();
    let writer = Keys::open(&log);
    let other = Keys::open(&log);
    let before = other.refreshed();
    assert_eq!(before.1, (0, vec![], [None, None]));
    let (parked_tx, parked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let w = &writer;
    let (points, samples, stats_at, last) = thread::scope(|s| {
        let committer = s.spawn(move || {
            STAGED_PARK.set(Some(Box::new(move |point| {
                parked_tx.send(point).unwrap();
                release_rx
                    .recv_timeout(PARK_TIMEOUT)
                    .expect("the reader releases the batch");
            })));
            let mut staged = w.store.staged_batch().unwrap();
            staged.set(w.x, 1).unwrap();
            staged.insert(&w.c, 10).unwrap();
            let mut pending = PendingWrite::new();
            pending.upsert(&w.k, 1, "a".to_owned());
            pending.upsert(&w.k, 2, "b".to_owned());
            staged.write_pending(pending).unwrap();
            staged.set(w.x, 2).unwrap();
            let last = staged.commit();
            STAGED_PARK.set(None);
            last.unwrap()
        });
        let (mut points, mut samples, mut stats) = (Vec::new(), Vec::new(), None);
        loop {
            let point = match parked_rx.recv_timeout(PARK_TIMEOUT) {
                Ok(point) => point,
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => panic!("the batch neither parks nor ends"),
            };
            points.push(point);
            samples.push((w.seen(), other.refreshed()));
            if point == StagedPark::Landed {
                let waiting = s.spawn(|| w.store.stats().unwrap().position);
                thread::sleep(Duration::from_millis(100));
                assert!(
                    !waiting.is_finished(),
                    "stats read the store between the run landing and its position moving"
                );
                stats = Some(waiting);
            }
            release_tx.send(()).unwrap();
        }
        let last = committer.join().unwrap();
        let stats_at = stats.expect("the commit parks once landed").join().unwrap();
        (points, samples, stats_at, last)
    });
    use StagedPark::{Applied, Landed};
    assert_eq!(
        points,
        [Applied, Applied, Applied, Applied, Applied, Landed]
    );
    let done = writer.seen();
    assert_eq!(
        done,
        (2, vec![10], [Some("a".to_owned()), Some("b".to_owned())])
    );
    assert_eq!(stats_at, last);
    for pair in samples.windows(2) {
        let (earlier, later) = (progress(&pair[0].0), progress(&pair[1].0));
        assert!(
            earlier.0 <= later.0 && earlier.1 <= later.1 && earlier.2 <= later.2,
            "the writer's values went back from {earlier:?} to {later:?}"
        );
    }
    for (_, observed) in &samples {
        assert!(
            *observed == before || *observed == (last, done.clone()),
            "the other store saw {observed:?}"
        );
    }
    assert_eq!(samples.last().unwrap().1, (last, done.clone()));
    drop(writer);
    drop(other);
    let reopened = Keys::open(&log);
    assert_eq!(reopened.store.stats().unwrap().position, last);
    assert_eq!(reopened.seen(), done);
}
