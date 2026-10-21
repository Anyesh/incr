//! The memo across real restarts. Each case runs this test binary twice
//! more: child `a` builds and checkpoints a memo over a `FileBackend` log
//! and a `FileMemo` and is killed with SIGKILL, then child `b` reopens both
//! and runs its own script. Its reads are compared with a cold run, with an
//! empty memo that never checkpoints, over a copy of the log `a` left.
//! Children talk to the parent through lines on stdout prefixed with
//! [`PREFIX`], and `b` hands back a [`Report`] file.

#![cfg(unix)]

use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use incr_core::persist_hooks::assert_invariants;
use incr_core::{Incr, IncrCollection, Runtime};
use incr_persist::memo::{
    BuildId, FileMemo, Memo, MemoBackend, MemoBuilder, MemoCells, MemoCtx, MemoOptions, MemoTable,
    MemoryMemo,
};
use incr_persist::{
    Durability, FileBackend, Input, Local, Options, PersistedCollection, Shared, Store,
};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::{Config, TestCaseError, TestRunner};
use serde::{Deserialize, Serialize};

const ROLE: &str = "INCR_MEMO_RESTART_ROLE";
const DIR: &str = "INCR_MEMO_RESTART_DIR";
const PREFIX: &str = "incr-memo-restart: ";
const READY: &str = "ready";
const SIGKILL: i32 = 9;

/// Not a hang bound: a child that outlives it has hung or crawled, and the
/// test fails instead of blocking the suite.
const CHILD_TIMEOUT: Duration = Duration::from_secs(60);

const BUILD: u128 = 0x5e57a27;

/// Random churn steps child `a` takes before it reports ready.
const CHURN_WARMUP: u32 = 12;

/// Concurrent checkpoints child `a` completes before it reports ready.
const CHURN_CHECKPOINTS: u64 = 3;

/// Equal to its other-signed zero, as `f64` is, but serialized with the
/// sign, so a collection's stored representative decides its encoding.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct F(f64);

impl Eq for F {}

impl Hash for F {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bits = if self.0 == 0.0 { 0 } else { self.0.to_bits() };
        bits.hash(state);
    }
}

/// Equal by `id` alone, though `note` is serialized and read too.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Tag {
    id: u8,
    note: u8,
}

impl PartialEq for Tag {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Tag {}

impl Hash for Tag {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// Floats an input or element takes: both zeros, so a write can be equal
/// to the held value and still change its bytes.
const FLOATS: [f64; 4] = [0.0, -0.0, 1.0, -2.5];

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Spec {
    /// Inputs `x0..`, each defaulting to 0.0. Churn never writes `x0`, so
    /// entries over it alone survive a kill unchanged.
    inputs: u8,
    /// Externals `e0..`, which only scripts set.
    externals: u8,
    /// Whether nodes read the `floats` and `tags` collections, which only a
    /// `Shared` closure can capture.
    collections: bool,
    /// Node `i` lives in table `i % 2` under key `i` and reads only nodes
    /// above `i`, so a graph is at most `nodes.len()` deep.
    nodes: Vec<Node>,
    a: Vec<Op>,
    /// When set, `a` keeps writing, reading and checkpointing after its
    /// script, on three threads under `threads`, and is killed while it
    /// does.
    churn: Option<Churn>,
    b: Vec<Op>,
    /// `b` runs its writes on one thread and its reads on two others.
    threads: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Node {
    /// Keeps only the value mod 3, so recomputes often cut off.
    coarse: bool,
    deps: Vec<Dep>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Dep {
    Input(u8),
    External(u8),
    Node(u32),
    /// Reads `even` or `odd` by the parity of a mix of the input's bits,
    /// so the dep list changes with the input.
    Branch {
        input: u8,
        even: u32,
        odd: u32,
    },
    Floats,
    Tags,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Op {
    Read(u32),
    /// Deletes the node and its dependents, so the next read restores it
    /// from the memo's index in the same process.
    Drop(u32),
    Set(u8, f64),
    External(u8, u128),
    Insert(F),
    Delete(F),
    Replace(Option<F>, F),
    InsertTag(Tag),
    DeleteTag(Tag),
    ReplaceTag(Option<Tag>, Tag),
    Checkpoint,
    /// Compacts the log.
    Compact,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Churn {
    seed: u64,
    kill_after_ms: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Report {
    /// The script's reads in order; empty for a threaded `b`.
    reads: Vec<u64>,
    /// Every node, read after the script.
    finals: Vec<u64>,
    runs: u64,
    reused: u64,
    revalidated: u64,
    corrupt: u64,
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn options() -> MemoOptions {
    MemoOptions::new(BuildId::Fixed(BUILD))
}

fn input_key(i: u8) -> String {
    format!("x{i}")
}

fn external_name(i: u8) -> String {
    format!("e{i}")
}

/// What a closure reads from the collections; only `Shared` captures them.
trait Flavor: MemoCells {
    type View: Send + Sync + 'static;

    fn view(
        floats: &PersistedCollection<F, Self>,
        tags: &PersistedCollection<Tag, Self>,
    ) -> Self::View;

    fn floats(view: &Self::View, rt: &Runtime<Self>) -> u64;

    fn tags(view: &Self::View, rt: &Runtime<Self>) -> u64;

    fn run_threaded(world: &World<Self>, ops: &[Op]);

    fn churn_threaded(world: &World<Self>, churn: Churn) -> !;
}

impl Flavor for Local {
    type View = ();

    fn view(_: &PersistedCollection<F, Self>, _: &PersistedCollection<Tag, Self>) {}

    fn floats(_: &(), _: &Runtime<Self>) -> u64 {
        unreachable!("a Local graph reads no collection")
    }

    fn tags(_: &(), _: &Runtime<Self>) -> u64 {
        unreachable!("a Local graph reads no collection")
    }

    fn run_threaded(_: &World<Self>, _: &[Op]) {
        unreachable!("Local runs on one thread")
    }

    fn churn_threaded(_: &World<Self>, _: Churn) -> ! {
        unreachable!("Local runs on one thread")
    }
}

impl Flavor for Shared {
    type View = (IncrCollection<F, Shared>, IncrCollection<Tag, Shared>);

    fn view(
        floats: &PersistedCollection<F, Self>,
        tags: &PersistedCollection<Tag, Self>,
    ) -> Self::View {
        (floats.handle().clone(), tags.handle().clone())
    }

    fn floats(view: &Self::View, rt: &Runtime<Self>) -> u64 {
        view.0
            .values(rt)
            .iter()
            .fold(0, |sum, f| sum.wrapping_add(mix(f.0.to_bits())))
    }

    fn tags(view: &Self::View, rt: &Runtime<Self>) -> u64 {
        view.1.values(rt).iter().fold(0, |sum, t| {
            sum.wrapping_add(mix(u64::from(t.id) << 8 | u64::from(t.note)))
        })
    }

    fn run_threaded(world: &World<Self>, ops: &[Op]) {
        let reads: Vec<u32> = ops
            .iter()
            .filter_map(|op| match op {
                Op::Read(j) => Some(*j),
                _ => None,
            })
            .collect();
        thread::scope(|s| {
            s.spawn(|| {
                for op in ops.iter().filter(|op| !matches!(op, Op::Read(_))) {
                    world.apply(op, false);
                }
            });
            s.spawn(|| {
                reads.iter().for_each(|&j| {
                    world.read(j);
                })
            });
            s.spawn(|| {
                reads.iter().rev().for_each(|&j| {
                    world.read(j);
                })
            });
        });
    }

    fn churn_threaded(world: &World<Self>, churn: Churn) -> ! {
        let checkpoints = AtomicU64::new(0);
        thread::scope(|s| {
            s.spawn(|| {
                let mut rng = churn.seed;
                loop {
                    if let Some(op) = churn_write(&world.spec, &mut rng) {
                        world.apply(&op, false);
                    }
                }
            });
            s.spawn(|| {
                let mut rng = !churn.seed;
                loop {
                    world.read(pick(&mut rng, world.spec.nodes.len()) as u32);
                }
            });
            s.spawn(|| loop {
                world.memo.checkpoint().unwrap();
                checkpoints.fetch_add(1, SeqCst);
            });
            while checkpoints.load(SeqCst) < CHURN_CHECKPOINTS {
                thread::sleep(Duration::from_millis(1));
            }
            say(READY);
            loop {
                thread::park();
            }
        })
    }
}

/// What both tables' closures capture: constants and handles.
struct Graph<C: Flavor> {
    spec: Arc<Spec>,
    inputs: Vec<Incr<f64>>,
    view: C::View,
    tables: [MemoTable<u32, u64>; 2],
    runs: Arc<AtomicU64>,
}

impl<C: Flavor> Graph<C> {
    fn eval(&self, ctx: &MemoCtx<'_, C>, i: u32) -> u64 {
        self.runs.fetch_add(1, SeqCst);
        let node = &self.spec.nodes[i as usize];
        let mut acc = mix(u64::from(i));
        for dep in &node.deps {
            let value = match *dep {
                Dep::Input(j) => ctx.rt().get(self.inputs[usize::from(j)]).to_bits(),
                Dep::External(j) => {
                    let fp = ctx.external(&external_name(j));
                    (fp as u64) ^ ((fp >> 64) as u64)
                }
                Dep::Node(j) => ctx.get(self.tables[j as usize % 2], j),
                Dep::Branch { input, even, odd } => {
                    let bits = ctx.rt().get(self.inputs[usize::from(input)]).to_bits();
                    let j = if mix(bits).is_multiple_of(2) {
                        even
                    } else {
                        odd
                    };
                    ctx.get(self.tables[j as usize % 2], j)
                }
                Dep::Floats => C::floats(&self.view, ctx.rt()),
                Dep::Tags => C::tags(&self.view, ctx.rt()),
            };
            acc = mix(acc ^ value);
        }
        if node.coarse {
            acc % 3
        } else {
            acc
        }
    }
}

/// A store and memo over one log, with the spec's inputs, collections and
/// node tables.
struct World<C: Flavor> {
    spec: Arc<Spec>,
    store: Arc<Store<C, FileBackend>>,
    memo: Memo<C, FileBackend>,
    inputs: Vec<Input<f64>>,
    floats: PersistedCollection<F, C>,
    tags: PersistedCollection<Tag, C>,
    tables: [MemoTable<u32, u64>; 2],
    runs: Arc<AtomicU64>,
}

impl<C: Flavor> World<C> {
    fn open(spec: &Arc<Spec>, log: &Path, memo: impl MemoBackend + 'static) -> Self {
        let store_options = Options {
            durability: Durability::Buffered,
            ..Options::default()
        };
        let store = Arc::new(Store::open(FileBackend::open(log).unwrap(), store_options).unwrap());
        let inputs: Vec<Input<f64>> = (0..spec.inputs)
            .map(|i| store.input(&input_key(i), 0.0).unwrap())
            .collect();
        let floats = store.collection::<F>("floats").unwrap();
        let tags = store.collection::<Tag>("tags").unwrap();
        let mut builder = MemoBuilder::<C>::new(options());
        let tables = [
            builder.table::<u32, u64>("even", 1),
            builder.table::<u32, u64>("odd", 1),
        ];
        let runs = Arc::new(AtomicU64::new(0));
        let graph = Arc::new(Graph::<C> {
            spec: Arc::clone(spec),
            inputs: inputs.iter().map(|x| x.handle()).collect(),
            view: C::view(&floats, &tags),
            tables,
            runs: Arc::clone(&runs),
        });
        for table in tables {
            let graph = Arc::clone(&graph);
            builder.define(table, move |ctx, &i| graph.eval(ctx, i));
        }
        let memo = builder.open(&store, memo).unwrap();
        World {
            spec: Arc::clone(spec),
            store,
            memo,
            inputs,
            floats,
            tags,
            tables,
            runs,
        }
    }

    fn node(&self, j: u32) -> Incr<u64> {
        self.memo.get_or_insert(self.tables[j as usize % 2], j)
    }

    fn read(&self, j: u32) -> u64 {
        let q = self.node(j);
        self.memo.runtime().get(q)
    }

    /// Applies `op`, returning a read's value. A cold world never
    /// checkpoints, so its memo stays empty.
    fn apply(&self, op: &Op, cold: bool) -> Option<u64> {
        let store = &*self.store;
        match *op {
            Op::Read(j) => return Some(self.read(j)),
            Op::Drop(j) => {
                if let Some(q) = self.memo.get(self.tables[j as usize % 2], &j) {
                    self.memo.runtime().delete_with_dependents(q);
                }
            }
            Op::Set(i, x) => drop(store.set(self.inputs[usize::from(i)], x).unwrap()),
            Op::External(i, fp) => drop(self.memo.set_external(&external_name(i), fp)),
            Op::Insert(f) => drop(self.floats.insert(store, f).unwrap()),
            Op::Delete(f) => drop(self.floats.delete(store, &f).unwrap()),
            Op::Replace(old, new) => drop(self.floats.replace(store, old.as_ref(), new).unwrap()),
            Op::InsertTag(t) => drop(self.tags.insert(store, t).unwrap()),
            Op::DeleteTag(t) => drop(self.tags.delete(store, &t).unwrap()),
            Op::ReplaceTag(old, new) => drop(self.tags.replace(store, old.as_ref(), new).unwrap()),
            Op::Checkpoint if cold => {}
            Op::Checkpoint => drop(self.memo.checkpoint().unwrap()),
            Op::Compact => drop(store.checkpoint().unwrap()),
        }
        None
    }

    /// Runs `b`'s script, reads every node and checks (I) and edge
    /// completeness at quiescence.
    fn run_b(&self, cold: bool) -> Report {
        let reads = if self.spec.threads && !cold {
            // Externals the script sets first are set before any restore.
            let ops = &self.spec.b;
            let first = ops
                .iter()
                .take_while(|op| matches!(op, Op::External(..)))
                .count();
            for op in &ops[..first] {
                self.apply(op, false);
            }
            C::run_threaded(self, &ops[first..]);
            Vec::new()
        } else {
            self.spec
                .b
                .iter()
                .filter_map(|op| self.apply(op, cold))
                .collect()
        };
        let finals = (0..self.spec.nodes.len() as u32)
            .map(|j| self.read(j))
            .collect();
        assert_invariants(self.memo.runtime());
        let stats = self.memo.stats();
        Report {
            reads,
            finals,
            runs: self.runs.load(SeqCst),
            reused: stats.reused,
            revalidated: stats.revalidated,
            corrupt: stats.corrupt,
        }
    }

    fn churn(&self, churn: Churn) -> ! {
        if self.spec.threads {
            C::churn_threaded(self, churn);
        }
        let mut rng = churn.seed;
        for step in 1.. {
            let op = match pick(&mut rng, 4) {
                0 => Some(Op::Read(pick(&mut rng, self.spec.nodes.len()) as u32)),
                1 => Some(Op::Checkpoint),
                _ => churn_write(&self.spec, &mut rng),
            };
            if let Some(op) = op {
                self.apply(&op, false);
            }
            if step == CHURN_WARMUP {
                say(READY);
            }
        }
        unreachable!()
    }
}

fn pick(rng: &mut u64, n: usize) -> usize {
    *rng = mix(*rng);
    (*rng % n as u64) as usize
}

/// A random mapped write for churn; `None` when the spec has nothing churn
/// may write.
fn churn_write(spec: &Spec, rng: &mut u64) -> Option<Op> {
    let float = |rng: &mut u64| element(FLOATS[pick(rng, FLOATS.len())]);
    let tag = |rng: &mut u64| element_tag((pick(rng, 3) as u8, pick(rng, 3) as u8));
    let kinds = if spec.collections { 5 } else { 1 };
    Some(match pick(rng, kinds) {
        0 if spec.inputs > 1 => {
            let i = 1 + pick(rng, usize::from(spec.inputs) - 1) as u8;
            Op::Set(i, FLOATS[pick(rng, FLOATS.len())])
        }
        0 => return None,
        1 => Op::Insert(float(rng)),
        2 => Op::Replace(Some(float(rng)), float(rng)),
        3 => Op::Delete(float(rng)),
        _ => Op::ReplaceTag(Some(tag(rng)), tag(rng)),
    })
}

// Writes to stdout directly, because the harness captures what `println!`
// prints in a test and would hold the message back until the test ends.
fn say(message: &str) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{PREFIX}{message}").unwrap();
    out.flush().unwrap();
}

fn spec_path(dir: &Path) -> PathBuf {
    dir.join("spec")
}

fn report_path(dir: &Path) -> PathBuf {
    dir.join("report")
}

fn a_reads_path(dir: &Path) -> PathBuf {
    dir.join("a-reads")
}

/// Plays `role` if this process is a child, returning whether it was.
fn play_child<C: Flavor>() -> bool {
    let Ok(role) = env::var(ROLE) else {
        return false;
    };
    let dir = PathBuf::from(env::var_os(DIR).expect("a child is given the case directory"));
    let spec: Arc<Spec> =
        Arc::new(postcard::from_bytes(&fs::read(spec_path(&dir)).unwrap()).unwrap());
    let world = World::<C>::open(
        &spec,
        &dir.join("log"),
        FileMemo::open(dir.join("memo")).unwrap(),
    );
    match role.as_str() {
        "a" => {
            // Killed with SIGKILL, so the parent learns of a panic only
            // through this message and the abort's signal.
            let default = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                default(info);
                say(&format!("panicked: {info}"));
                std::process::abort();
            }));
            let reads: Vec<u64> = spec
                .a
                .iter()
                .filter_map(|op| world.apply(op, false))
                .collect();
            fs::write(a_reads_path(&dir), postcard::to_allocvec(&reads).unwrap()).unwrap();
            if let Some(churn) = spec.churn {
                world.churn(churn);
            }
            say(READY);
            // The parent kills this process; it never writes to stdin.
            let mut line = String::new();
            std::io::stdin().lock().read_line(&mut line).unwrap();
        }
        "b" => {
            let report = world.run_b(false);
            fs::write(report_path(&dir), postcard::to_allocvec(&report).unwrap()).unwrap();
        }
        other => panic!("unknown child role {other:?}"),
    }
    true
}

/// A re-executed copy of this test binary playing `role` in `dir`.
struct Peer {
    child: Child,
    _stdin: ChildStdin,
    messages: Receiver<String>,
    stderr: JoinHandle<String>,
}

impl Peer {
    fn spawn(test: &str, role: &str, dir: &Path) -> Peer {
        let mut child = Command::new(env::current_exe().unwrap())
            .args([test, "--exact", "--test-threads=1", "--nocapture"])
            .env(ROLE, role)
            .env(DIR, dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (send, messages) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { return };
                // The harness prints a test's name without a newline before
                // running it, so a message can start partway through a line.
                if let Some((_, message)) = line.split_once(PREFIX) {
                    if send.send(message.to_owned()).is_err() {
                        return;
                    }
                }
            }
        });
        let mut stderr = child.stderr.take().unwrap();
        let stderr = thread::spawn(move || {
            let mut text = String::new();
            let _ = std::io::Read::read_to_string(&mut stderr, &mut text);
            text
        });
        Peer {
            _stdin: child.stdin.take().unwrap(),
            child,
            messages,
            stderr,
        }
    }

    fn fail(mut self, what: &str) -> ! {
        let _ = self.child.kill();
        let status = self.child.wait().unwrap();
        let stderr = self.stderr.join().unwrap();
        panic!("child {what} ({status}):\n{stderr}");
    }

    fn kill_when_ready(mut self, delay: Duration) {
        match self.messages.recv_timeout(CHILD_TIMEOUT) {
            Ok(message) if message == READY => {}
            Ok(other) => self.fail(&format!("sent {other:?}")),
            Err(RecvTimeoutError::Timeout) => self.fail("never became ready"),
            Err(RecvTimeoutError::Disconnected) => self.fail("exited before it was ready"),
        }
        thread::sleep(delay);
        self.child.kill().unwrap();
        let status = self.child.wait().unwrap();
        if status.signal() != Some(SIGKILL) {
            self.fail(&format!("was not killed but exited with {status}"));
        }
        loop {
            match self.messages.recv_timeout(CHILD_TIMEOUT) {
                Ok(message) if message == READY => {}
                Ok(other) => self.fail(&format!("sent {other:?}")),
                Err(RecvTimeoutError::Timeout) => self.fail("held stdout open after it was killed"),
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    }

    fn finish(mut self) {
        let deadline = Instant::now() + CHILD_TIMEOUT;
        loop {
            match self.child.try_wait().unwrap() {
                Some(status) if status.success() => return,
                Some(_) => self.fail("failed"),
                None if Instant::now() >= deadline => self.fail("hung"),
                None => thread::sleep(Duration::from_millis(5)),
            }
        }
    }
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Runs `a`, kills it, runs `b` on what it left, and returns `b`'s report,
/// a cold run's over a copy of the same log, and `a`'s script reads.
fn run_case<C: Flavor>(test: &str, spec: &Spec) -> (Report, Report, Vec<u64>) {
    let dir = tempfile::tempdir().unwrap();
    let case = dir.path().join("case");
    fs::create_dir_all(&case).unwrap();
    fs::write(spec_path(&case), postcard::to_allocvec(spec).unwrap()).unwrap();
    let delay = spec.churn.map_or(0, |churn| churn.kill_after_ms);
    Peer::spawn(test, "a", &case).kill_when_ready(Duration::from_millis(delay));
    let cold_log = dir.path().join("cold");
    copy_dir(&case.join("log"), &cold_log);
    Peer::spawn(test, "b", &case).finish();
    let b: Report = postcard::from_bytes(&fs::read(report_path(&case)).unwrap()).unwrap();
    let spec = Arc::new(spec.clone());
    let cold = World::<C>::open(&spec, &cold_log, MemoryMemo::new()).run_b(true);
    let a_reads = postcard::from_bytes(&fs::read(a_reads_path(&case)).unwrap()).unwrap();
    (b, cold, a_reads)
}

/// `b` matches the cold run: values always, reads in order and at most the
/// cold closure runs when it ran on one thread. A kill never leaves a
/// frame `b` finds corrupt.
fn check(spec: &Spec, b: &Report, cold: &Report) -> Result<(), TestCaseError> {
    prop_assert_eq!(&b.finals, &cold.finals, "final values");
    prop_assert_eq!(b.corrupt, 0);
    if !spec.threads {
        prop_assert_eq!(&b.reads, &cold.reads, "script reads");
        prop_assert!(
            b.runs <= cold.runs,
            "{} closure runs after the restart, {} cold",
            b.runs,
            cold.runs
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Shape {
    inputs: u8,
    externals: u8,
    nodes: u32,
    collections: bool,
    drops: bool,
}

type RawDep = (u8, u32, u32);
type RawOp = (u8, u32, f64, f64, (u8, u8), (u8, u8), bool);

fn resolve_dep(shape: Shape, i: u32, (kind, a, b): RawDep) -> Dep {
    let later = shape.nodes - i - 1;
    let above = |r: u32| i + 1 + r % later;
    let input = |r: u32| (r % u32::from(shape.inputs)) as u8;
    match kind % 7 {
        0 | 1 if later > 0 => Dep::Node(above(a)),
        2 if later > 0 => Dep::Node(i + 1),
        3 if later > 0 => Dep::Branch {
            input: input(a),
            even: above(a / 3),
            odd: above(b),
        },
        4 if shape.externals > 0 => Dep::External((a % u32::from(shape.externals)) as u8),
        5 if shape.collections => {
            if a % 2 == 0 {
                Dep::Floats
            } else {
                Dep::Tags
            }
        }
        _ => Dep::Input(input(a)),
    }
}

/// A random collection element. Equal elements that encode differently,
/// -0.0 and 0.0 or tags that differ only in their note, give a restart a
/// representative that depends on hash order until the board item
/// T10-collection-rep-order lands, so random ops draw one encoding per
/// class and the pinned edge tests carry the rest.
fn element(x: f64) -> F {
    F(if x == 0.0 { 0.0 } else { x })
}

fn element_tag((id, _): (u8, u8)) -> Tag {
    Tag { id, note: 0 }
}

fn resolve_op(shape: Shape, (kind, a, x, y, t, u, some): RawOp) -> Op {
    let node = a % shape.nodes;
    let tag = element_tag;
    match kind % 14 {
        0..=2 => Op::Read(node),
        3 if shape.drops => Op::Drop(node),
        3 | 4 => Op::Checkpoint,
        5 => Op::Compact,
        6 if shape.externals > 0 => {
            Op::External((a % u32::from(shape.externals)) as u8, u128::from(a % 3))
        }
        7 if shape.collections => Op::Insert(element(x)),
        8 if shape.collections => Op::Delete(element(x)),
        9 if shape.collections => Op::Replace(some.then_some(element(x)), element(y)),
        10 if shape.collections => Op::InsertTag(tag(t)),
        11 if shape.collections => Op::ReplaceTag(some.then_some(tag(t)), tag(u)),
        12 if shape.collections => Op::DeleteTag(tag(t)),
        _ => Op::Set((a % u32::from(shape.inputs)) as u8, x),
    }
}

fn spec_strategy(collections: bool, threads: bool) -> impl Strategy<Value = Spec> {
    let float = || prop::sample::select(FLOATS.to_vec());
    let tag = || (0u8..3, 0u8..3);
    let raw_op = move || {
        (
            any::<u8>(),
            any::<u32>(),
            float(),
            float(),
            tag(),
            tag(),
            any::<bool>(),
        )
    };
    let raw_node = (
        any::<bool>(),
        vec((any::<u8>(), any::<u32>(), any::<u32>()), 0..4),
    );
    let churn = (any::<u64>(), 0u64..20).prop_map(|(seed, kill_after_ms)| Churn {
        seed,
        kill_after_ms,
    });
    let churn = if threads {
        churn.prop_map(Some).boxed()
    } else {
        proptest::option::weighted(0.3, churn).boxed()
    };
    (
        1u8..=3,
        0u8..=2,
        1u32..=48,
        vec(raw_node, 48),
        vec(raw_op(), 0..24),
        vec(raw_op(), 0..24),
        churn,
    )
        .prop_map(move |(inputs, externals, nodes, raw_nodes, a, b, churn)| {
            let shape = Shape {
                inputs,
                externals,
                nodes,
                collections,
                drops: true,
            };
            let nodes_spec = raw_nodes
                .into_iter()
                .take(nodes as usize)
                .enumerate()
                .map(|(i, (coarse, deps))| Node {
                    coarse,
                    deps: deps
                        .into_iter()
                        .map(|raw| resolve_dep(shape, i as u32, raw))
                        .collect(),
                })
                .collect();
            let b_shape = Shape {
                drops: !threads,
                ..shape
            };
            Spec {
                inputs,
                externals,
                collections,
                nodes: nodes_spec,
                a: a.into_iter().map(|raw| resolve_op(shape, raw)).collect(),
                churn,
                b: b.into_iter().map(|raw| resolve_op(b_shape, raw)).collect(),
                threads,
            }
        })
}

fn chain(len: u32, deps: impl Fn(u32) -> Vec<Dep>) -> Vec<Node> {
    (0..len)
        .map(|i| Node {
            coarse: false,
            deps: deps(i),
        })
        .collect()
}

fn reads(nodes: u32) -> impl Iterator<Item = Op> {
    (0..nodes).map(Op::Read)
}

/// Node 0 reads node 1, which reads `dep`. `a` reads both and checkpoints
/// after each write, so its last checkpoint holds values computed before
/// its last write took effect in the runtime; `b` reads both.
fn edge_spec(dep: Dep, writes: Vec<Op>, collections: bool) -> Spec {
    let mut a = Vec::new();
    for write in writes {
        a.push(write);
        a.extend(reads(2));
        a.push(Op::Checkpoint);
    }
    Spec {
        inputs: 1,
        externals: 0,
        collections,
        nodes: chain(2, |i| {
            if i == 0 {
                vec![Dep::Node(1)]
            } else {
                vec![dep.clone()]
            }
        }),
        a,
        churn: None,
        b: reads(2).collect(),
        threads: false,
    }
}

/// With `as_a`, `b` must also read what `a` read last: a restart rebuilds a
/// collection with the representatives `a` held. Not so for an input, whose
/// runtime keeps 0.0 through an equal write of -0.0 that the log records.
/// Repeated, because a restart that rebuilds a collection in hash order
/// differs only by chance.
fn edge_case<C: Flavor>(test: &str, spec: Spec, as_a: bool) {
    if play_child::<C>() {
        return;
    }
    for round in 0..6 {
        let (b, cold, a_reads) = run_case::<C>(test, &spec);
        if let Err(e) = check(&spec, &b, &cold) {
            panic!("round {round}: {e}");
        }
        if as_a {
            assert_eq!(
                b.reads,
                a_reads[a_reads.len() - 2..],
                "round {round}: reads after the restart"
            );
        }
    }
}

fn minus_zero_input(collections: bool) -> Spec {
    let writes = vec![Op::Set(0, 0.0), Op::Set(0, -0.0)];
    edge_spec(Dep::Input(0), writes, collections)
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_input_set_to_minus_zero_after_zero_restarts_like_a_cold_run_local() {
    edge_case::<Local>(
        "an_input_set_to_minus_zero_after_zero_restarts_like_a_cold_run_local",
        minus_zero_input(false),
        false,
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_input_set_to_minus_zero_after_zero_restarts_like_a_cold_run_shared() {
    edge_case::<Shared>(
        "an_input_set_to_minus_zero_after_zero_restarts_like_a_cold_run_shared",
        minus_zero_input(true),
        false,
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_zero_element_replaced_by_minus_zero_restarts_like_a_cold_run() {
    let writes = vec![Op::Insert(F(0.0)), Op::Replace(Some(F(0.0)), F(-0.0))];
    edge_case::<Shared>(
        "a_zero_element_replaced_by_minus_zero_restarts_like_a_cold_run",
        edge_spec(Dep::Floats, writes, true),
        true,
    );
}

/// The collection keeps -0.0, the first inserted, as the representative
/// of both occurrences.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "T10-collection-rep-order: a restart folds a collection's elements in hash order"]
fn zero_elements_of_both_signs_restart_like_a_cold_run() {
    let writes = vec![Op::Insert(F(-0.0)), Op::Insert(F(0.0))];
    edge_case::<Shared>(
        "zero_elements_of_both_signs_restart_like_a_cold_run",
        edge_spec(Dep::Floats, writes, true),
        true,
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_element_replaced_by_an_equal_one_with_another_note_restarts_like_a_cold_run() {
    let old = Tag { id: 1, note: 0 };
    let writes = vec![
        Op::InsertTag(old),
        Op::ReplaceTag(Some(old), Tag { id: 1, note: 2 }),
    ];
    edge_case::<Shared>(
        "an_element_replaced_by_an_equal_one_with_another_note_restarts_like_a_cold_run",
        edge_spec(Dep::Tags, writes, true),
        true,
    );
}

/// Node 0 reads node 1 when `x0` is 0.0 or 1.0 and node 2 when it is
/// -2.5; node 1 reads `x0`. `a` checkpoints both over 0.0, then sets -2.5.
/// `b`'s first read reruns node 0, which no longer reads node 1, so node 1
/// must not run before it: a cold run doesn't run it until `x0` is 1.0.
fn moved_branch() -> Spec {
    let parity = |x: f64| mix(f64::to_bits(x)) % 2;
    assert_eq!(parity(0.0), parity(1.0));
    assert_ne!(parity(0.0), parity(-2.5));
    let (even, odd) = if parity(0.0) == 0 { (1, 2) } else { (2, 1) };
    let deps = [
        vec![Dep::Branch {
            input: 0,
            even,
            odd,
        }],
        vec![Dep::Input(0)],
        vec![],
    ];
    Spec {
        inputs: 1,
        externals: 0,
        collections: false,
        nodes: chain(3, |i| deps[i as usize].clone()),
        a: vec![Op::Read(0), Op::Checkpoint, Op::Set(0, -2.5)],
        churn: None,
        b: vec![Op::Read(0), Op::Set(0, 1.0)],
        threads: false,
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_restored_entry_whose_branch_moved_runs_no_more_closures_than_cold_local() {
    edge_case::<Local>(
        "a_restored_entry_whose_branch_moved_runs_no_more_closures_than_cold_local",
        moved_branch(),
        false,
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_restored_entry_whose_branch_moved_runs_no_more_closures_than_cold_shared() {
    edge_case::<Shared>(
        "a_restored_entry_whose_branch_moved_runs_no_more_closures_than_cold_shared",
        moved_branch(),
        false,
    );
}

fn oracle<C: Flavor>(test: &str, collections: bool, threads: bool, cases: u32) {
    let reused = AtomicUsize::new(0);
    let saved = AtomicUsize::new(0);
    let run = |spec: &Spec| -> Result<(), TestCaseError> {
        let (b, cold, _) = run_case::<C>(test, spec);
        check(spec, &b, &cold)?;
        if b.reused > 0 {
            reused.fetch_add(1, SeqCst);
        }
        if b.runs < cold.runs {
            saved.fetch_add(1, SeqCst);
        }
        Ok(())
    };
    let mut runner = TestRunner::new(Config {
        cases,
        max_shrink_iters: 64,
        failure_persistence: None,
        ..Config::default()
    });
    if let Err(e) = runner.run(&spec_strategy(collections, threads), |spec| run(&spec)) {
        panic!("{e}");
    }
    // Values equal to a cold run's prove nothing if nothing was reused.
    assert!(reused.load(SeqCst) > 0, "no case reused an entry");
    if !threads {
        assert!(
            saved.load(SeqCst) > 0,
            "no case ran fewer closures than cold"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn restart_oracle_local() {
    if !play_child::<Local>() {
        oracle::<Local>("restart_oracle_local", false, false, 48);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn restart_oracle_shared() {
    if !play_child::<Shared>() {
        oracle::<Shared>("restart_oracle_shared", true, false, 48);
    }
}

/// Values and (I) only: concurrent reads can see a set in flight, which
/// can rerun a closure.
#[test]
#[cfg_attr(miri, ignore)]
fn restart_oracle_shared_threads() {
    if !play_child::<Shared>() {
        oracle::<Shared>("restart_oracle_shared_threads", true, true, 24);
    }
}

/// Nodes 0..8 read the churned inputs `x1` and `x2` and the collections;
/// nodes 8..16 read only `x0`, `e0` and each other, which churn never
/// writes. `a` checkpoints on one thread while two others set inputs,
/// write the collections and read, and is killed while they do; `b` must
/// match a cold run and still reuse the untouched half.
#[test]
#[cfg_attr(miri, ignore)]
fn checkpoints_under_concurrent_mapped_sets_restore_like_a_cold_run() {
    const TEST: &str = "checkpoints_under_concurrent_mapped_sets_restore_like_a_cold_run";
    if play_child::<Shared>() {
        return;
    }
    let nodes = chain(16, |i| match i {
        0..=7 => vec![
            Dep::Input(1 + (i % 2) as u8),
            Dep::Node(i + 1),
            if i % 3 == 0 { Dep::Floats } else { Dep::Tags },
            Dep::Branch {
                input: 2,
                even: 8 + i,
                odd: 15,
            },
        ],
        8..=14 => vec![Dep::Input(0), Dep::External(0), Dep::Node(i + 1)],
        _ => vec![Dep::Input(0)],
    });
    for (round, kill_after_ms) in [0, 2, 5, 10, 0, 3, 7, 25].into_iter().enumerate() {
        let mut a = vec![Op::External(0, 7), Op::Set(0, 2.5), Op::Insert(F(1.0))];
        a.extend(reads(16));
        a.push(Op::Checkpoint);
        let spec = Spec {
            inputs: 3,
            externals: 1,
            collections: true,
            nodes: nodes.clone(),
            a,
            churn: Some(Churn {
                seed: round as u64,
                kill_after_ms,
            }),
            b: std::iter::once(Op::External(0, 7))
                .chain(reads(16))
                .collect(),
            threads: true,
        };
        let (b, cold, _) = run_case::<Shared>(TEST, &spec);
        assert_eq!(b.finals, cold.finals, "round {round}");
        assert_eq!(b.corrupt, 0, "round {round}");
        assert!(b.reused >= 8, "round {round} reused only {}", b.reused);
    }
}

/// After a count-1 `replace(0.0 -> -0.0)` the collection's fp is that of
/// the new representative: an entry over it, removed and restored in the
/// same process, recomputes, and once checkpointed it is reused by a
/// restart whose open computes the fp from the log.
#[test]
fn an_equal_replace_moves_the_collection_fp_to_the_new_representative() {
    let dir = tempfile::tempdir().unwrap();
    let runs = Arc::new(AtomicU64::new(0));
    let open = || {
        let backend = FileBackend::open(dir.path().join("log")).unwrap();
        let store = Arc::new(Store::<Shared, _>::open(backend, Options::default()).unwrap());
        let floats = store.collection::<F>("floats").unwrap();
        let view = floats.handle().clone();
        let mut builder = MemoBuilder::new(options());
        let negatives = builder.table::<(), u64>("negatives", 1);
        let r = Arc::clone(&runs);
        builder.define(negatives, move |ctx, _| {
            r.fetch_add(1, SeqCst);
            view.values(ctx.rt())
                .iter()
                .filter(|f| f.0.is_sign_negative())
                .count() as u64
        });
        let memo = FileMemo::open(dir.path().join("memo")).unwrap();
        let memo = builder.open(&store, memo).unwrap();
        (store, floats, memo, negatives)
    };
    {
        let (store, floats, memo, negatives) = open();
        floats.insert(&store, F(0.0)).unwrap();
        let q = memo.get_or_insert(negatives, ());
        assert_eq!(memo.runtime().get(q), 0);
        memo.checkpoint().unwrap();
        floats.replace(&store, Some(&F(0.0)), F(-0.0)).unwrap();
        memo.runtime().delete_node(q);
        let q = memo.get_or_insert(negatives, ());
        assert_eq!(memo.runtime().get(q), 1);
        assert_eq!(runs.load(SeqCst), 2);
        let stats = memo.stats();
        assert_eq!((stats.reused, stats.revalidated), (0, 1));
        memo.checkpoint().unwrap();
    }
    let (_store, _floats, memo, negatives) = open();
    let q = memo.get_or_insert(negatives, ());
    assert_eq!(memo.runtime().get(q), 1);
    assert_eq!(runs.load(SeqCst), 2);
    assert_eq!(memo.stats().reused, 1);
}
