//! `MemoOptions::checkpoint_after_entries`: `checkpoint_if_due` on any
//! runtime, and the automatic-checkpoint thread on `Shared`.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use incr_persist::memo::{
    BuildId, EncodedEntry, GenHeader, GenId, Generation, IndexItem, Memo, MemoBackend, MemoBuilder,
    MemoCells, MemoOptions, MemoTable, MemoryMemo,
};
use incr_persist::{Input, Local, MemoryBackend, Options, Result, Shared, Store};

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

both!(checkpoint_if_due_without_a_trigger_does_nothing);

const BUILD: u128 = 0x7e1;

struct World<C: MemoCells> {
    store: Arc<Store<C, MemoryBackend>>,
    memo: Memo<C, MemoryBackend>,
    x: Input<u64>,
    t: MemoTable<u32, u64>,
}

fn options() -> MemoOptions {
    MemoOptions::new(BuildId::Fixed(BUILD))
}

impl<C: MemoCells> World<C> {
    fn open(options: MemoOptions) -> Self {
        Self::open_on(options, MemoryMemo::new())
    }

    fn open_on(options: MemoOptions, backend: impl MemoBackend + 'static) -> Self {
        let store = Arc::new(Store::open(MemoryBackend::new(), Options::default()).unwrap());
        let x = store.input("x", 1_u64).unwrap();
        let xh = x.handle();
        let mut builder = MemoBuilder::new(options);
        let t = builder.table::<u32, u64>("t", 1);
        builder.define(t, move |ctx, &k| ctx.rt().get(xh) + u64::from(k));
        let memo = builder.open(&store, backend).unwrap();
        World { store, memo, x, t }
    }

    fn read(&self, k: u32) -> u64 {
        let q = self.memo.get_or_insert(self.t, k);
        self.memo.runtime().get(q)
    }

    fn wait_for_checkpoints(&self, n: u64) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while self.memo.stats().checkpoints < n {
            assert!(Instant::now() < deadline, "no automatic checkpoint ran");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// Local only: on Shared the trigger also wakes the automatic thread,
/// which races these checkpoints.
#[test]
fn checkpoint_if_due_runs_once_the_trigger_is_reached() {
    let w = World::<Local>::open(options().checkpoint_after_entries(2));
    w.read(1);
    assert!(w.memo.checkpoint_if_due().unwrap().is_none());
    w.read(2);
    let done = w
        .memo
        .checkpoint_if_due()
        .unwrap()
        .expect("two entries are due");
    assert_eq!(done.written, 2);
    let stats = w.memo.stats();
    assert_eq!((stats.checkpoints, stats.unsaved_entries), (1, 0));
    assert!(w.memo.checkpoint_if_due().unwrap().is_none());
    w.store.set(w.x, 5).unwrap();
    w.read(1);
    assert!(w.memo.checkpoint_if_due().unwrap().is_none());
    w.read(2);
    assert!(w.memo.checkpoint_if_due().unwrap().is_some());
}

fn checkpoint_if_due_without_a_trigger_does_nothing<C: MemoCells>() {
    let w = World::<C>::open(options());
    for k in 0..10 {
        w.read(k);
    }
    assert!(w.memo.checkpoint_if_due().unwrap().is_none());
    assert_eq!(w.memo.stats().checkpoints, 0);
}

#[test]
fn reads_never_checkpoint_on_local() {
    let w = World::<Local>::open(options().checkpoint_after_entries(1));
    for k in 0..10 {
        w.read(k);
    }
    let stats = w.memo.stats();
    assert_eq!((stats.checkpoints, stats.unsaved_entries), (0, 10));
}

#[test]
fn the_trigger_wakes_the_automatic_checkpoint_thread() {
    let w = World::<Shared>::open(options().checkpoint_after_entries(3));
    w.read(1);
    w.read(2);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(w.memo.stats().checkpoints, 0);
    w.read(3);
    w.wait_for_checkpoints(1);
    // The trigger fires inside the third compute, which may land after
    // the checkpoint's snapshot; it then still counts as unsaved.
    let stats = w.memo.stats();
    assert_eq!(stats.entries + stats.unsaved_entries, 3);
}

#[test]
fn the_trigger_wakes_the_thread_before_its_interval() {
    let w = World::<Shared>::open(
        options()
            .auto_checkpoint_secs(3600)
            .checkpoint_after_entries(2),
    );
    w.read(1);
    w.read(2);
    w.wait_for_checkpoints(1);
}

#[test]
fn an_interval_with_nothing_unsaved_writes_nothing() {
    let w = World::<Shared>::open(options().auto_checkpoint_secs(1));
    w.read(1);
    w.wait_for_checkpoints(1);
    std::thread::sleep(Duration::from_millis(2500));
    assert_eq!(w.memo.stats().checkpoints, 1);
    w.store.set(w.x, 2).unwrap();
    w.read(1);
    w.wait_for_checkpoints(2);
}

/// A trigger reached while an unlogged scope is open is skipped, not
/// counted as an error; the scope's close wakes the thread again, with no
/// further compute.
#[test]
fn the_trigger_wakes_the_thread_again_after_a_skipped_checkpoint() {
    let w = World::<Shared>::open(options().checkpoint_after_entries(2));
    let scope = w.store.unlogged_scope().unwrap();
    w.read(1);
    w.read(2);
    std::thread::sleep(Duration::from_millis(200));
    let stats = w.memo.stats();
    assert_eq!(stats.checkpoints, 0);
    assert_eq!(stats.auto_checkpoint_errors, 0);
    scope.close().unwrap();
    w.wait_for_checkpoints(1);
}

/// A [`MemoryMemo`] whose first generation write waits for [`Self::release`].
#[derive(Clone, Default)]
struct HeldWrite {
    inner: MemoryMemo,
    state: Arc<(Mutex<(u32, bool)>, Condvar)>,
}

impl HeldWrite {
    fn wait_for_first_write(&self) {
        let (lock, cond) = &*self.state;
        let guard = lock.lock().unwrap();
        drop(cond.wait_while(guard, |(writes, _)| *writes == 0).unwrap());
    }

    fn release(&self) {
        let (lock, cond) = &*self.state;
        lock.lock().unwrap().1 = true;
        cond.notify_all();
    }
}

impl MemoBackend for HeldWrite {
    type Guard = <MemoryMemo as MemoBackend>::Guard;

    fn lock_checkpoint(&self) -> Result<Self::Guard> {
        self.inner.lock_checkpoint()
    }

    fn load(&self) -> Result<Option<Generation>> {
        self.inner.load()
    }

    fn read_entry(&self, generation: GenId, item: &IndexItem) -> Result<Option<Vec<u8>>> {
        self.inner.read_entry(generation, item)
    }

    fn write_generation(
        &self,
        base: Option<GenId>,
        id: GenId,
        header: &GenHeader,
        entries: &[EncodedEntry],
    ) -> Result<Generation> {
        let (lock, cond) = &*self.state;
        let mut state = lock.lock().unwrap();
        state.0 += 1;
        cond.notify_all();
        if state.0 == 1 {
            drop(cond.wait_while(state, |(_, released)| !*released).unwrap());
        } else {
            drop(state);
        }
        self.inner.write_generation(base, id, header, entries)
    }

    fn remove_generation(&self, generation: GenId) -> Result<()> {
        self.inner.remove_generation(generation)
    }
}

/// Computes that pass the trigger while the woken checkpoint is writing
/// find the thread not yet armed; it must still run for them once that
/// checkpoint ends, with no further compute.
#[test]
fn entries_due_during_a_checkpoint_are_saved_without_another_compute() {
    let backend = HeldWrite::default();
    let w = World::<Shared>::open_on(options().checkpoint_after_entries(2), backend.clone());
    w.read(1);
    w.read(2);
    backend.wait_for_first_write();
    w.read(3);
    w.read(4);
    assert_eq!(w.memo.stats().unsaved_entries, 2);
    backend.release();
    w.wait_for_checkpoints(2);
}
