//! Workload bodies shared by the criterion benches and the `icount`
//! driver, so the code criterion times and the code perf counts are the
//! same.

use criterion::{BatchSize, Bencher};
use incr_core::{Cells, ErasedIncr, Incr, IncrCollection, Runtime, Shared};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::hint::black_box;
use std::sync::Barrier;

/// One measured operation. `setup` builds what a single `run` consumes;
/// criterion times only `run` and drops its output untimed, which is
/// where a workload that rebuilds per iteration tears down. The icount
/// driver counts setup, run and teardown alike, since perf stat sees the
/// whole process.
pub trait Workload {
    type Input;
    type Output<'a>
    where
        Self: 'a;
    fn setup(&self) -> Self::Input;
    fn run(&self, input: Self::Input) -> Self::Output<'_>;
}

pub fn steady<W: Workload<Input = ()>>(b: &mut Bencher, w: &W) {
    b.iter(|| w.run(()));
}

pub fn batched<W: Workload>(b: &mut Bencher, w: &W) {
    b.iter_batched(|| w.setup(), |input| w.run(input), BatchSize::SmallInput);
}

/// Deletes `root` and everything that depends on or is owned by it when
/// dropped, returning the runtime to the size it had before the setup
/// that built `root`, so per-iteration rebuilds recycle the same slots.
pub struct Teardown<'a, C: Cells> {
    rt: &'a Runtime<C>,
    root: ErasedIncr,
}

impl<C: Cells> Drop for Teardown<'_, C> {
    fn drop(&mut self) {
        self.rt.delete_with_dependents(self.root);
    }
}

pub struct Chain<C: Cells> {
    rt: Runtime<C>,
    input: Incr<i64>,
    output: Incr<i64>,
    val: Cell<i64>,
}

impl<C: Cells> Chain<C> {
    pub fn new(n: usize) -> Self {
        let rt: Runtime<C> = Runtime::new();
        let input = rt.create_input(1_i64);
        let mut prev = input;
        for _ in 0..n {
            let dep = prev;
            prev = rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
        }
        let _ = rt.get(prev);
        Self {
            rt,
            input,
            output: prev,
            val: Cell::new(1),
        }
    }
}

impl<C: Cells> Workload for Chain<C> {
    type Input = ();
    type Output<'a> = i64;
    fn setup(&self) {}
    fn run(&self, (): ()) -> i64 {
        let val = self.val.get().wrapping_add(1);
        self.val.set(val);
        self.rt.set(self.input, val);
        black_box(self.rt.get(self.output))
    }
}

pub struct Diamond<C: Cells> {
    rt: Runtime<C>,
    input: Incr<i64>,
    out: Incr<i64>,
    val: Cell<i64>,
}

impl<C: Cells> Diamond<C> {
    pub fn new() -> Self {
        let rt: Runtime<C> = Runtime::new();
        let input = rt.create_input(1_i64);
        let a = rt.create_query(move |rt| rt.get(input).wrapping_add(10));
        let b = rt.create_query(move |rt| rt.get(input).wrapping_add(100));
        let out = rt.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b)));
        let _ = rt.get(out);
        Self {
            rt,
            input,
            out,
            val: Cell::new(1),
        }
    }
}

impl<C: Cells> Workload for Diamond<C> {
    type Input = ();
    type Output<'a> = i64;
    fn setup(&self) {}
    fn run(&self, (): ()) -> i64 {
        let val = self.val.get().wrapping_add(1);
        self.val.set(val);
        self.rt.set(self.input, val);
        black_box(self.rt.get(self.out))
    }
}

pub struct EarlyCutoff<C: Cells> {
    rt: Runtime<C>,
    input: Incr<i64>,
    after: Incr<i64>,
    val: Cell<i64>,
}

impl<C: Cells> EarlyCutoff<C> {
    pub fn new() -> Self {
        let rt: Runtime<C> = Runtime::new();
        let input = rt.create_input(200_i64);
        let clamped = rt.create_query(move |rt| rt.get(input).min(100));
        let after = rt.create_query(move |rt| rt.get(clamped).wrapping_add(1));
        let _ = rt.get(after);
        Self {
            rt,
            input,
            after,
            val: Cell::new(200),
        }
    }
}

impl<C: Cells> Workload for EarlyCutoff<C> {
    type Input = ();
    type Output<'a> = i64;
    fn setup(&self) {}
    fn run(&self, (): ()) -> i64 {
        // Always above 100, so the clamp yields 100 and cuts off.
        let val = self.val.get().wrapping_add(1);
        self.val.set(val);
        self.rt.set(self.input, val);
        black_box(self.rt.get(self.after))
    }
}

/// A steady-state collection of `size` elements under a pipeline ending
/// in `count`; each run inserts one new element and reads the count.
pub struct InsertThenRead<C: Cells> {
    rt: Runtime<C>,
    col: IncrCollection<i64, C>,
    count: Incr<u64>,
    next: Cell<i64>,
}

impl<C: Cells> InsertThenRead<C> {
    pub fn pipeline(size: usize) -> Self {
        let rt: Runtime<C> = Runtime::new();
        let col: IncrCollection<i64, C> = rt.create_collection();
        let evens = col.filter(&rt, |x| x % 2 == 0);
        let doubled = evens.map(&rt, |x| x * 2);
        let count = doubled.count(&rt);
        Self::populate(rt, col, count, size)
    }

    pub fn simple_count(size: usize) -> Self {
        let rt: Runtime<C> = Runtime::new();
        let col: IncrCollection<i64, C> = rt.create_collection();
        let count = col.count(&rt);
        Self::populate(rt, col, count, size)
    }

    fn populate(
        rt: Runtime<C>,
        col: IncrCollection<i64, C>,
        count: Incr<u64>,
        size: usize,
    ) -> Self {
        for i in 0..size as i64 {
            col.insert(&rt, i);
        }
        let _ = rt.get(count);
        Self {
            rt,
            col,
            count,
            next: Cell::new(size as i64),
        }
    }
}

impl<C: Cells> Workload for InsertThenRead<C> {
    type Input = ();
    type Output<'a> = u64;
    fn setup(&self) {}
    fn run(&self, (): ()) -> u64 {
        let next = self.next.get();
        self.col.insert(&self.rt, next);
        self.next.set(next + 1);
        black_box(self.rt.get(self.count))
    }
}

/// The from-scratch baseline for [`InsertThenRead::pipeline`]: insert
/// into a plain `HashSet`, then rerun filter, map and count over all of
/// it.
pub struct BatchRebuild {
    elements: RefCell<HashSet<i64>>,
    next: Cell<i64>,
}

impl BatchRebuild {
    pub fn new(size: usize) -> Self {
        Self {
            elements: RefCell::new((0..size as i64).collect()),
            next: Cell::new(size as i64),
        }
    }
}

impl Workload for BatchRebuild {
    type Input = ();
    type Output<'a> = usize;
    fn setup(&self) {}
    // The map is kept, though `count` ignores it, because this must stay
    // the same filter, map and count shape as the incremental pipeline.
    #[allow(clippy::suspicious_map)]
    fn run(&self, (): ()) -> usize {
        let mut elements = self.elements.borrow_mut();
        let next = self.next.get();
        elements.insert(next);
        self.next.set(next + 1);
        black_box(
            elements
                .iter()
                .filter(|x| *x % 2 == 0)
                .map(|x| x * 2)
                .count(),
        )
    }
}

/// A `get` of a Clean query.
pub struct GetClean<C: Cells> {
    rt: Runtime<C>,
    query: Incr<i64>,
}

impl<C: Cells> GetClean<C> {
    pub fn new() -> Self {
        let rt: Runtime<C> = Runtime::new();
        let input = rt.create_input(1_i64);
        let query = rt.create_query(move |rt| rt.get(input).wrapping_add(1));
        let _ = rt.get(query);
        Self { rt, query }
    }
}

impl<C: Cells> Workload for GetClean<C> {
    type Input = ();
    type Output<'a> = i64;
    fn setup(&self) {}
    fn run(&self, (): ()) -> i64 {
        black_box(self.rt.get(self.query))
    }
}

/// `READERS` threads, the calling one among them, each reading one Clean
/// query `READS` times after meeting at a barrier. Threads are spawned
/// per run because reader threads parked between runs would spin or
/// sleep for a time no iteration count controls, and perf counts every
/// thread; `READS` is large enough that the spawns are a small share.
pub struct GetContended {
    rt: Runtime<Shared>,
    query: Incr<i64>,
}

impl GetContended {
    pub const READERS: usize = 4;
    pub const READS: usize = 10_000;

    pub fn new() -> Self {
        let rt: Runtime<Shared> = Runtime::new();
        let input = rt.create_input(1_i64);
        let query = rt.create_query(move |rt| rt.get(input).wrapping_add(1));
        let _ = rt.get(query);
        Self { rt, query }
    }

    fn read_all(&self, barrier: &Barrier) {
        barrier.wait();
        for _ in 0..Self::READS {
            black_box(self.rt.get(self.query));
        }
    }
}

impl Workload for GetContended {
    type Input = ();
    type Output<'a> = ();
    fn setup(&self) {}
    fn run(&self, (): ()) {
        let barrier = Barrier::new(Self::READERS);
        std::thread::scope(|s| {
            for _ in 1..Self::READERS {
                s.spawn(|| self.read_all(&barrier));
            }
            self.read_all(&barrier);
        });
    }
}

/// A query reading `width` queries over one input. Each setup builds a
/// fresh copy, so the measured depth-0 `get` is the top query's first
/// compute and each of its reads is a nested first compute.
pub struct Fanout<C: Cells> {
    rt: Runtime<C>,
    width: i64,
}

impl<C: Cells> Fanout<C> {
    pub fn new(width: i64) -> Self {
        Self {
            rt: Runtime::new(),
            width,
        }
    }
}

impl<C: Cells> Workload for Fanout<C> {
    type Input = (Incr<i64>, Incr<i64>);
    type Output<'a> = Teardown<'a, C>;
    fn setup(&self) -> (Incr<i64>, Incr<i64>) {
        let input = self.rt.create_input(1_i64);
        let leaves: Vec<Incr<i64>> = (0..self.width)
            .map(|i| {
                self.rt
                    .create_query(move |rt| rt.get(input).wrapping_add(i))
            })
            .collect();
        let top = self.rt.create_query(move |rt| {
            leaves
                .iter()
                .fold(0_i64, |acc, &leaf| acc.wrapping_add(rt.get(leaf)))
        });
        (input, top)
    }
    fn run(&self, (input, top): (Incr<i64>, Incr<i64>)) -> Teardown<'_, C> {
        black_box(self.rt.get(top));
        Teardown {
            rt: &self.rt,
            root: input.erase(),
        }
    }
}

/// `delete_node` of a lone input.
pub struct DeleteSingle<C: Cells> {
    rt: Runtime<C>,
}

impl<C: Cells> DeleteSingle<C> {
    pub fn new() -> Self {
        Self { rt: Runtime::new() }
    }
}

impl<C: Cells> Workload for DeleteSingle<C> {
    type Input = Incr<i64>;
    type Output<'a> = ();
    fn setup(&self) -> Incr<i64> {
        self.rt.create_input(0_i64)
    }
    fn run(&self, node: Incr<i64>) {
        self.rt.delete_node(node);
    }
}

/// `delete_with_dependents` of an input under `CHAINS` chains of `LEN`
/// queries, each reading the one below. Setup computes every chain from
/// the bottom up, so no compute nests a non-Clean read.
pub struct DeleteCascade<C: Cells> {
    rt: Runtime<C>,
}

impl<C: Cells> DeleteCascade<C> {
    pub const CHAINS: usize = 10;
    pub const LEN: usize = 100;

    pub fn new() -> Self {
        Self { rt: Runtime::new() }
    }
}

impl<C: Cells> Workload for DeleteCascade<C> {
    type Input = Incr<i64>;
    type Output<'a> = usize;
    fn setup(&self) -> Incr<i64> {
        let input = self.rt.create_input(1_i64);
        for _ in 0..Self::CHAINS {
            let mut prev = input;
            for _ in 0..Self::LEN {
                let dep = prev;
                prev = self.rt.create_query(move |rt| rt.get(dep).wrapping_add(1));
                black_box(self.rt.get(prev));
            }
        }
        input
    }
    fn run(&self, input: Incr<i64>) -> usize {
        black_box(self.rt.delete_with_dependents(input))
    }
}

/// `delete_node` of a computed `group_by` router owning `GROUPS` groups,
/// which the strict delete takes with it. The source collection's
/// version node is torn down after.
pub struct DeleteOwnedTree<C: Cells> {
    rt: Runtime<C>,
}

impl<C: Cells> DeleteOwnedTree<C> {
    pub const GROUPS: i64 = 100;

    pub fn new() -> Self {
        Self { rt: Runtime::new() }
    }
}

impl<C: Cells> Workload for DeleteOwnedTree<C> {
    type Input = (Incr<u64>, Incr<u64>);
    type Output<'a> = Teardown<'a, C>;
    fn setup(&self) -> (Incr<u64>, Incr<u64>) {
        let col: IncrCollection<i64, C> = self.rt.create_collection();
        for i in 0..Self::GROUPS {
            col.insert(&self.rt, i);
        }
        let router = col.group_by(&self.rt, |x| *x).version_node();
        black_box(self.rt.get(router));
        (col.version_node(), router)
    }
    fn run(&self, (source, router): (Incr<u64>, Incr<u64>)) -> Teardown<'_, C> {
        self.rt.delete_node(router);
        Teardown {
            rt: &self.rt,
            root: source.erase(),
        }
    }
}
