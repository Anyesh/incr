//! Stores in separate OS processes sharing one log. Each test re-executes
//! this test binary filtered to itself, with `INCR_PERSIST_CHILD` naming
//! the role the child plays and `INCR_PERSIST_DIR` the log's location.
//! Children talk to the parent through lines on stdout prefixed with
//! [`PREFIX`], since the test harness prints its own lines there too, and
//! the parent talks back through the child's stdin.

use std::env;
use std::io::{BufRead, BufReader, Lines, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use incr_persist::memo::{
    decode_entry, write_checkpoint, CheckpointInput, Entry, FileMemo, MemoBackend, StoredDep,
    TableDesc, DEFAULT_MAX_BYTES,
};
use incr_persist::{
    Backend, Error, FileBackend, Local, Options, Position, ReadOutcome, Record, Store,
};

const CHILD: &str = "INCR_PERSIST_CHILD";
const DIR: &str = "INCR_PERSIST_DIR";
const PREFIX: &str = "incr-persist-child: ";

fn child_role() -> Option<String> {
    env::var(CHILD).ok()
}

fn child_dir() -> PathBuf {
    PathBuf::from(env::var_os(DIR).expect("a child is given the log location"))
}

// Writes to stdout directly, because the harness captures what `println!`
// prints in a test and would hold the message back until the test ends.
fn say(message: &str) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{PREFIX}{message}").unwrap();
    out.flush().unwrap();
}

struct Peer {
    child: Child,
    lines: Lines<BufReader<ChildStdout>>,
}

impl Peer {
    fn spawn(test: &str, role: &str, dir: &Path) -> Peer {
        let mut child = Command::new(env::current_exe().unwrap())
            .args([test, "--exact", "--test-threads=1"])
            .env(CHILD, role)
            .env(DIR, dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let lines = BufReader::new(child.stdout.take().unwrap()).lines();
        Peer { child, lines }
    }

    /// The next message from the child, skipping the harness's own output.
    fn expect(&mut self) -> String {
        for line in &mut self.lines {
            // The harness prints a test's name without a newline before
            // running it, so a message can start partway through a line.
            if let Some((_, message)) = line.unwrap().split_once(PREFIX) {
                return message.to_owned();
            }
        }
        let status = self.child.wait().unwrap();
        panic!("the child exited ({status}) before sending the next message");
    }

    fn send(&mut self, message: &str) {
        let stdin = self.child.stdin.as_mut().unwrap();
        writeln!(stdin, "{message}").unwrap();
        stdin.flush().unwrap();
    }

    fn finish(mut self) {
        drop(self.child.stdin.take());
        for line in &mut self.lines {
            line.unwrap();
        }
        let status = self.child.wait().unwrap();
        assert!(status.success(), "the child failed: {status}");
    }
}

#[cfg(feature = "sqlite")]
mod sqlite {
    use std::thread;
    use std::time::{Duration, Instant};

    use incr_persist::{SqliteBackend, SqliteDurability, SqliteOptions};

    use super::*;

    const WRITES: u32 = 200;

    // These tests check that writers progress and take fair turns, not how
    // fast they do: on a loaded box a writer can wait seconds for the turns
    // ahead of it, and the default timeout would fail it for that alone.
    const BUSY_TIMEOUT: Duration = Duration::from_secs(120);

    fn open(path: &Path) -> SqliteBackend {
        SqliteBackend::open_with(
            path,
            SqliteOptions {
                busy_timeout: BUSY_TIMEOUT,
                ..SqliteOptions::default()
            },
        )
        .unwrap()
    }

    fn view(store: &Store<Local, SqliteBackend>) -> String {
        let events = store.collection::<(u8, u32)>("events").unwrap();
        let shared = store.input("shared", 0u32).unwrap();
        let first = store.input("own/0", 0u32).unwrap();
        let second = store.input("own/1", 0u32).unwrap();
        let rt = store.runtime();
        let mut values = events.handle().values(rt);
        values.sort();
        assert_eq!(values.len() as u32, 2 * (WRITES - WRITES / 20));
        format!(
            "events={values:?} shared={} own={},{}",
            rt.get(shared.handle()),
            rt.get(first.handle()),
            rt.get(second.handle())
        )
    }

    /// Both processes run this with a different `writer`: interleaved
    /// writes to keys they share and keys of their own, then refreshes
    /// until the other's `done` flag shows its writes have all landed.
    fn write_and_converge(path: &Path, writer: u8) -> String {
        let store = Store::<Local, _>::open(open(path), Options::default()).unwrap();
        let events = store.collection::<(u8, u32)>("events").unwrap();
        let shared = store.input("shared", 0u32).unwrap();
        let own = store.input(&format!("own/{writer}"), 0u32).unwrap();
        for i in 0..WRITES {
            events.insert(&store, (writer, i)).unwrap();
            store.set(shared, i).unwrap();
            if i % 20 == 0 {
                events.delete(&store, &(writer, i)).unwrap();
                store.delete(own).unwrap();
            } else {
                store.set(own, i).unwrap();
            }
        }
        let done = store.input(&format!("done/{writer}"), false).unwrap();
        store.set(done, true).unwrap();

        let other = store.input(&format!("done/{}", 1 - writer), false).unwrap();
        let deadline = Instant::now() + Duration::from_secs(300);
        while !store.runtime().get(other.handle()) {
            assert!(Instant::now() < deadline, "the other writer never finished");
            thread::sleep(Duration::from_millis(5));
            store.refresh().unwrap();
        }
        view(&store)
    }

    #[test]
    fn two_sqlite_writer_processes_converge_after_refresh() {
        if let Some(role) = child_role() {
            if role == "sqlite-writer" {
                let path = child_dir().join("log.sqlite");
                open(&path);
                say("ready");
                say(&format!("state {}", write_and_converge(&path, 1)));
            }
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.sqlite");
        SqliteBackend::open(&path).unwrap();
        let mut peer = Peer::spawn(
            "sqlite::two_sqlite_writer_processes_converge_after_refresh",
            "sqlite-writer",
            dir.path(),
        );
        assert_eq!(peer.expect(), "ready");
        let mine = write_and_converge(&path, 0);
        let theirs = peer.expect();
        assert_eq!(theirs.strip_prefix("state "), Some(mine.as_str()));
        peer.finish();

        let restarted = Store::<Local, _>::open(open(&path), Options::default()).unwrap();
        assert_eq!(
            view(&restarted),
            mine,
            "a fresh open agrees with both processes"
        );
    }

    const CONTENDERS: usize = 3;
    const CONTENTION_WINDOW: Duration = Duration::from_secs(4);

    /// Opens the log, runs `wait_for_start`, then appends in a tight loop
    /// for the window. Returns the appends that succeeded, the ones that
    /// failed, and the first failure.
    fn contend(path: &Path, wait_for_start: impl FnOnce()) -> (u64, u64, String) {
        let backend = SqliteBackend::open_with(
            path,
            SqliteOptions {
                durability: SqliteDurability::Full,
                busy_timeout: BUSY_TIMEOUT,
                ..SqliteOptions::default()
            },
        )
        .unwrap();
        let record = Record::Set {
            key: "contended".into(),
            value: vec![0; 16],
        };
        wait_for_start();
        let start = Instant::now();
        let (mut appended, mut failed, mut first_error) = (0, 0, String::new());
        while start.elapsed() < CONTENTION_WINDOW {
            match backend.append(std::slice::from_ref(&record)) {
                Ok(_) => appended += 1,
                Err(e) => {
                    if failed == 0 {
                        first_error = format!("{e:?}");
                    }
                    failed += 1;
                }
            }
        }
        (appended, failed, first_error)
    }

    /// Writers appending back to back under `synchronous=FULL` each leave
    /// the lock free for microseconds between commits that take
    /// milliseconds, so a writer that only retries on a timer keeps finding
    /// it held. Every writer must still get a fair share of turns.
    #[test]
    fn sqlite_writer_processes_appending_continuously_all_progress() {
        if let Some(role) = child_role() {
            if role == "sqlite-contender" {
                let path = child_dir().join("log.sqlite");
                let (appended, failed, first_error) = contend(&path, || {
                    say("ready");
                    let mut go = String::new();
                    std::io::stdin().read_line(&mut go).unwrap();
                });
                say(&format!("{appended} {failed} {first_error}"));
            }
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("log.sqlite");
        SqliteBackend::open(&path).unwrap();
        let mut peers: Vec<Peer> = (0..CONTENDERS)
            .map(|_| {
                Peer::spawn(
                    "sqlite::sqlite_writer_processes_appending_continuously_all_progress",
                    "sqlite-contender",
                    dir.path(),
                )
            })
            .collect();
        for peer in &mut peers {
            assert_eq!(peer.expect(), "ready");
        }
        for peer in &mut peers {
            peer.send("go");
        }
        let reports: Vec<(u64, u64, String)> = peers
            .iter_mut()
            .map(|peer| {
                let report = peer.expect();
                let mut fields = report.splitn(3, ' ');
                let appended = fields.next().unwrap().parse().unwrap();
                let failed = fields.next().unwrap().parse().unwrap();
                (appended, failed, fields.next().unwrap_or("").to_owned())
            })
            .collect();
        for peer in peers {
            peer.finish();
        }

        let total: u64 = reports.iter().map(|(appended, ..)| appended).sum();
        let counts: Vec<_> = reports
            .iter()
            .map(|(appended, failed, _)| (appended, failed))
            .collect();
        for (appended, failed, first_error) in &reports {
            assert_eq!(
                *failed, 0,
                "a writer failed {failed} appends ({first_error}); \
                 (appended, failed) per writer: {counts:?}"
            );
            assert!(
                *appended * 4 * CONTENDERS as u64 >= total,
                "a writer got under a quarter of an even share; \
                 (appended, failed) per writer: {counts:?}"
            );
        }
        let mut out = Vec::new();
        SqliteBackend::open_read_only(&path)
            .unwrap()
            .read_from(Position::ZERO, &mut out)
            .unwrap();
        assert_eq!(out.len() as u64, total);
    }
}

/// Counts the reads that came back `Compacted`, to show which way a reader
/// crossed a compaction.
struct Counting {
    inner: FileBackend,
    compacted: Arc<AtomicUsize>,
}

impl Backend for Counting {
    fn append(&self, records: &[Record]) -> incr_persist::Result<Position> {
        self.inner.append(records)
    }

    fn sync(&self, upto: Position) -> incr_persist::Result<()> {
        self.inner.sync(upto)
    }

    fn read_from(
        &self,
        after: Position,
        out: &mut Vec<(Position, Record)>,
    ) -> incr_persist::Result<ReadOutcome> {
        let outcome = self.inner.read_from(after, out)?;
        if let ReadOutcome::Compacted { .. } = outcome {
            self.compacted.fetch_add(1, Ordering::SeqCst);
        }
        Ok(outcome)
    }

    fn load_snapshot(&self) -> incr_persist::Result<Option<(Position, Vec<Record>)>> {
        self.inner.load_snapshot()
    }

    fn snapshot_id(&self) -> incr_persist::Result<Option<incr_persist::SnapshotId>> {
        self.inner.snapshot_id()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> incr_persist::Result<()> {
        self.inner.compact(upto, folded)
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }
}

/// What a store holds for the follow test's keys, with a query over them
/// so that recomputation in the reader is part of the comparison.
fn follow_view<B: Backend>(store: &Store<Local, B>) -> String {
    let x = store.input("x", 0u32).unwrap();
    let items = store.collection::<u32>("items").unwrap();
    let rt = store.runtime();
    let sum = items
        .handle()
        .reduce(rt, |elements: &[u32]| elements.iter().sum::<u32>());
    let x = x.handle();
    let total = rt.create_query(move |rt| rt.get(sum) + rt.get(x));
    let mut values = items.handle().values(rt);
    values.sort();
    format!("x={} items={values:?} total={}", rt.get(x), rt.get(total))
}

fn file_reader() {
    let compacted = Arc::new(AtomicUsize::new(0));
    let backend = Counting {
        inner: FileBackend::open_read_only(child_dir()).unwrap(),
        compacted: Arc::clone(&compacted),
    };
    let store = Store::<Local, _>::open(backend, Options::default()).unwrap();
    let x = store.input("x", 0u32).unwrap();
    let items = store.collection::<u32>("items").unwrap();
    let rt = store.runtime();
    let sum = items
        .handle()
        .reduce(rt, |elements: &[u32]| elements.iter().sum::<u32>());
    let total = {
        let x = x.handle();
        rt.create_query(move |rt| rt.get(sum) + rt.get(x))
    };
    say("ready");
    for line in std::io::stdin().lines() {
        assert_eq!(line.unwrap(), "check");
        store.refresh().unwrap();
        let mut values = items.handle().values(rt);
        values.sort();
        say(&format!(
            "x={} items={values:?} total={} compacted={}",
            rt.get(x.handle()),
            rt.get(total),
            compacted.load(Ordering::SeqCst)
        ));
    }
}

#[test]
fn a_file_reader_process_follows_a_writer_across_compactions() {
    if let Some(role) = child_role() {
        if role == "file-reader" {
            file_reader();
        }
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let writer =
        Store::<Local, _>::open(FileBackend::open(dir.path()).unwrap(), Options::default())
            .unwrap();
    let x = writer.input("x", 0u32).unwrap();
    let items = writer.collection::<u32>("items").unwrap();
    writer.set(x, 1).unwrap();
    for item in 1..=5 {
        items.insert(&writer, item).unwrap();
    }
    let mut peer = Peer::spawn(
        "a_file_reader_process_follows_a_writer_across_compactions",
        "file-reader",
        dir.path(),
    );
    assert_eq!(peer.expect(), "ready");

    let check = |peer: &mut Peer| -> usize {
        peer.send("check");
        let reply = peer.expect();
        let (state, compacted) = reply.rsplit_once(" compacted=").unwrap();
        assert_eq!(state, follow_view(&writer));
        compacted.parse().unwrap()
    };

    writer.set(x, 2).unwrap();
    items.insert(&writer, 10).unwrap();
    writer.checkpoint().unwrap();
    items.insert(&writer, 11).unwrap();
    items.delete(&writer, &1).unwrap();
    let after_one = check(&mut peer);

    writer.set(x, 3).unwrap();
    items.insert(&writer, 20).unwrap();
    writer.checkpoint().unwrap();
    items.insert(&writer, 21).unwrap();
    writer.checkpoint().unwrap();
    items.insert(&writer, 22).unwrap();
    items.delete(&writer, &2).unwrap();
    let after_two = check(&mut peer);
    assert!(
        after_two > after_one,
        "two compactions between reads leave the reader behind the snapshot"
    );

    writer.delete(x).unwrap();
    items.insert(&writer, 30).unwrap();
    assert_eq!(check(&mut peer), after_two);
    peer.finish();
}

fn memo_input(keys: std::ops::Range<u32>) -> CheckpointInput {
    let table = TableDesc {
        name: "t".into(),
        version: 1,
        key_type: "u32".into(),
        value_type: "u32".into(),
    };
    CheckpointInput {
        build: 1,
        tables: vec![table],
        fresh: keys
            .map(|key| Entry {
                table: 0,
                key: key.to_le_bytes().to_vec(),
                value: vec![key as u8; 64],
                deps: vec![StoredDep::External {
                    name: format!("x{key}"),
                    fp: u128::from(key),
                }],
                last_used: 0,
            })
            .collect(),
        max_bytes: DEFAULT_MAX_BYTES,
    }
}

fn memo_lock_holder() {
    let memo = FileMemo::open(child_dir()).unwrap();
    let guard = memo.lock_checkpoint().unwrap();
    say("locked");
    let mut lines = std::io::stdin().lines();
    assert_eq!(lines.next().unwrap().unwrap(), "release");
    drop(guard);
    say("released");
}

#[test]
fn a_memo_checkpoint_lock_held_by_another_process_is_busy() {
    if let Some(role) = child_role() {
        if role == "memo-lock-holder" {
            memo_lock_holder();
        }
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let memo = FileMemo::open(dir.path()).unwrap();
    let mut peer = Peer::spawn(
        "a_memo_checkpoint_lock_held_by_another_process_is_busy",
        "memo-lock-holder",
        dir.path(),
    );
    assert_eq!(peer.expect(), "locked");
    assert!(matches!(memo.lock_checkpoint(), Err(Error::Busy)));
    assert!(matches!(
        write_checkpoint(&memo, memo_input(0..3)),
        Err(Error::Busy)
    ));
    assert!(memo.load().unwrap().is_none());
    peer.send("release");
    assert_eq!(peer.expect(), "released");
    write_checkpoint(&memo, memo_input(0..3)).unwrap();
    peer.finish();
}

fn memo_checkpoint_loop() {
    let memo = FileMemo::open(child_dir()).unwrap();
    for round in 1.. {
        let done = write_checkpoint(&memo, memo_input(0..round * 50)).unwrap();
        if let Some(old) = done.replaced {
            memo.remove_generation(old).unwrap();
        }
        say(&format!("wrote {round}"));
    }
}

/// Kills a process that checkpoints in a loop at varying points, then
/// checks that the store holds a whole generation and takes the next
/// checkpoint.
#[test]
fn a_memo_checkpoint_killed_midway_leaves_a_whole_generation() {
    if let Some(role) = child_role() {
        if role == "memo-checkpoint-loop" {
            memo_checkpoint_loop();
        }
        return;
    }
    for delay_us in [0, 150, 400, 900, 2000] {
        let dir = tempfile::tempdir().unwrap();
        let mut peer = Peer::spawn(
            "a_memo_checkpoint_killed_midway_leaves_a_whole_generation",
            "memo-checkpoint-loop",
            dir.path(),
        );
        assert_eq!(peer.expect(), "wrote 1");
        std::thread::sleep(std::time::Duration::from_micros(delay_us));
        peer.child.kill().unwrap();
        peer.child.wait().unwrap();

        let memo = FileMemo::open(dir.path()).unwrap();
        let loaded = memo.load().unwrap().unwrap();
        assert!(!loaded.items().is_empty(), "a written generation is whole");
        assert_eq!(loaded.items().len() as u64, loaded.header().seq * 50);
        for item in loaded.items() {
            let frame = memo.read_entry(loaded.id(), item).unwrap().unwrap();
            decode_entry(&frame).unwrap();
        }
        let next = write_checkpoint(&memo, memo_input(0..7)).unwrap();
        assert_eq!(next.stats.corrupt, 0);
        assert_eq!(memo.load().unwrap().unwrap().id(), next.generation.id());
    }
}
