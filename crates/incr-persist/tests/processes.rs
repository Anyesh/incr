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

use incr_persist::{Backend, FileBackend, Local, Options, Position, ReadOutcome, Record, Store};

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

    use incr_persist::SqliteBackend;

    use super::*;

    const WRITES: u32 = 200;

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
        let store = Store::<Local, _>::open(SqliteBackend::open(path).unwrap(), Options::default())
            .unwrap();
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
        let deadline = Instant::now() + Duration::from_secs(60);
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
                SqliteBackend::open(&path).unwrap();
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

        let restarted =
            Store::<Local, _>::open(SqliteBackend::open(&path).unwrap(), Options::default())
                .unwrap();
        assert_eq!(
            view(&restarted),
            mine,
            "a fresh open agrees with both processes"
        );
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
