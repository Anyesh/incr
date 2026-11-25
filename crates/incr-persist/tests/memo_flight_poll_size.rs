//! A memo flight's `poll(Duration::ZERO)` does one unit of work whatever
//! the entry count: on a first generation, on a later one with a seed
//! prune, and when a cancel settles a pruning or committed flight. Work is
//! counted as what the polling thread asks of the allocator during one
//! call, since every step that grows with the entry count allocates or
//! frees in proportion to it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use incr_persist::memo::{
    BuildId, Memo, MemoBackend, MemoBuilder, MemoCheckpoint, MemoOptions, MemoTable, MemoryMemo,
};
use incr_persist::{Error, Input, Local, MemoryBackend, Options, Store};

struct Metered;

thread_local! {
    static METERING: Cell<bool> = const { Cell::new(false) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static FREES: Cell<usize> = const { Cell::new(0) };
}

fn count(bytes: usize, frees: usize) {
    let _ = METERING.try_with(|on| {
        if on.get() {
            BYTES.with(|b| b.set(b.get() + bytes));
            FREES.with(|f| f.set(f.get() + frees));
        }
    });
}

unsafe impl GlobalAlloc for Metered {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size(), 0);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size(), 0);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(0, 1);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size, 1);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Metered = Metered;

/// Entries per phase: each seed map, dirty shard and generation the
/// phases build is many units, so a step proportional to any of them
/// stands out from a unit's work.
const N: u32 = 1 << 16;
/// A unit's allowance: a finish slice of 4,096 entries or an inspect
/// batch of at most 4,096 records allocates up to about 9 MiB, as seed map
/// buckets grow, and frees up to about 12,500 times on this memo.
const MOST_BYTES: usize = 16 << 20;
const MOST_FREES: usize = 32 * 1024;
const WAIT: Duration = Duration::from_secs(120);

#[derive(Default)]
struct Most {
    bytes: usize,
    frees: usize,
    polls: usize,
}

impl Most {
    fn poll(
        &mut self,
        flight: &MemoCheckpoint,
        memo: &Memo<Local, MemoryBackend>,
    ) -> Result<bool, Error> {
        BYTES.with(|b| b.set(0));
        FREES.with(|f| f.set(0));
        METERING.with(|on| on.set(true));
        let polled = flight.poll(memo, Duration::ZERO);
        METERING.with(|on| on.set(false));
        self.bytes = self.bytes.max(BYTES.with(Cell::get));
        self.frees = self.frees.max(FREES.with(Cell::get));
        self.polls += 1;
        polled
    }

    fn within(&self, phase: &str) -> Option<String> {
        eprintln!(
            "{phase}: {} polls, most bytes {}, most frees {}",
            self.polls, self.bytes, self.frees
        );
        (self.bytes > MOST_BYTES || self.frees > MOST_FREES).then(|| {
            format!(
                "{phase}: a poll allocated {} bytes and freed {} times (a unit's allowance is {MOST_BYTES} and {MOST_FREES})",
                self.bytes, self.frees
            )
        })
    }
}

/// `node(0) = y` and `node(k) = node(k / 2) + k`: every entry but the
/// first has a query dep, so inspection copies seeds and records deps.
struct Fx {
    _store: Arc<Store<Local, MemoryBackend>>,
    memo: Memo<Local, MemoryBackend>,
    backend: MemoryMemo,
    _y: Input<u64>,
    node: MemoTable<u32, u64>,
}

impl Fx {
    // A memo opens over an `Arc` of its store, a Local one included.
    #[allow(clippy::arc_with_non_send_sync)]
    fn open() -> Fx {
        let store =
            Arc::new(Store::<Local, _>::open(MemoryBackend::new(), Options::default()).unwrap());
        let y = store.input("y", 1_u64).unwrap();
        let yh = y.handle();
        let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(0x241)));
        let node = builder.table::<u32, u64>("node", 1);
        builder.define(node, move |ctx, &k| {
            if k == 0 {
                ctx.rt().get(yh)
            } else {
                ctx.get(node, k / 2) + u64::from(k)
            }
        });
        let backend = MemoryMemo::new();
        let memo = builder.open(&store, backend.clone()).unwrap();
        Fx {
            _store: store,
            memo,
            backend,
            _y: y,
            node,
        }
    }

    fn compute(&self, keys: std::ops::Range<u32>) {
        for k in keys {
            let handle = self.memo.get_or_insert(self.node, k);
            self.memo.runtime().get(handle);
        }
    }

    fn tip(&self) -> Option<incr_persist::memo::GenId> {
        self.backend
            .load_chain(&[])
            .unwrap()
            .map(|chain| chain.tip().id())
    }

    fn run(&self, most: &mut Most) {
        let flight = self.memo.begin_checkpoint().unwrap();
        let deadline = Instant::now() + WAIT;
        while !most.poll(&flight, &self.memo).unwrap() {
            assert!(Instant::now() < deadline, "the flight did not finish");
        }
    }

    fn cancel_when(&self, most: &mut Most, ready: impl Fn() -> bool) {
        let flight = self.memo.begin_checkpoint().unwrap();
        let deadline = Instant::now() + WAIT;
        while !ready() {
            assert!(
                !most.poll(&flight, &self.memo).unwrap(),
                "finished before the cancel"
            );
            assert!(
                Instant::now() < deadline,
                "the flight never got ready to cancel"
            );
        }
        flight.cancel();
        loop {
            match most.poll(&flight, &self.memo) {
                Ok(false) => assert!(Instant::now() < deadline, "the cancel never settled"),
                Ok(true) => panic!("a cancelled flight finished"),
                Err(Error::Cancelled) => return,
                Err(error) => panic!("{error}"),
            }
        }
    }
}

#[test]
fn every_memo_flight_poll_does_at_most_one_unit() {
    let fx = Fx::open();
    let mut over = Vec::new();

    fx.compute(0..N);
    let mut first = Most::default();
    fx.run(&mut first);
    over.extend(first.within("first generation"));

    fx.compute(N..2 * N);
    let mut later = Most::default();
    fx.run(&mut later);
    over.extend(later.within("later generation, after a prune"));

    let mut pruning = Most::default();
    let polls = Cell::new(0);
    fx.cancel_when(&mut pruning, || polls.replace(polls.get() + 1) == 1);
    over.extend(pruning.within("cancelled while pruning"));

    fx.compute(2 * N..3 * N);
    let before = fx.tip();
    let mut committed = Most::default();
    fx.cancel_when(&mut committed, || fx.tip() != before);
    over.extend(committed.within("cancelled once committed"));

    assert!(over.is_empty(), "{over:#?}");
}
