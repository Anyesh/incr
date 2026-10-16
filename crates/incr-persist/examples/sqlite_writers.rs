//! Aggregate set throughput of several processes writing one SQLite log.
//!
//! ```text
//! cargo run --release -p incr-persist --features sqlite --example sqlite_writers -- [writers] [seconds] [full|buffered]
//! ```
//!
//! The parent re-executes this binary once per writer, starts them all
//! together, and has each write its own input for a fixed window, counting
//! the writes that finish inside it. Every write also applies the other
//! writers' records that landed before it, as any store sharing a log
//! does. The log lives in a fresh directory under the workspace's
//! `target/` (or `INCR_PERSIST_BENCH_DIR`), so fsync reaches a real disk.

use std::env;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use incr_persist::{
    Durability, Local, Options, SqliteBackend, SqliteDurability, SqliteOptions, Store,
};

const WRITER: &str = "INCR_PERSIST_WRITER";

struct Config {
    path: PathBuf,
    seconds: f64,
    durability: SqliteDurability,
}

fn open(config: &Config) -> Store<Local, SqliteBackend> {
    let backend = SqliteBackend::open_with(
        &config.path,
        SqliteOptions {
            durability: config.durability,
            busy_timeout: Duration::from_secs(30),
            ..SqliteOptions::default()
        },
    )
    .unwrap();
    let durability = match config.durability {
        SqliteDurability::Full => Durability::Sync,
        SqliteDurability::Buffered => Durability::Buffered,
    };
    Store::open(
        backend,
        Options {
            durability,
            ..Options::default()
        },
    )
    .unwrap()
}

fn writer(id: &str, config: &Config) {
    let store = open(config);
    let input = store.input(&format!("writer/{id}"), 0u64).unwrap();
    println!("ready");
    let mut go = String::new();
    std::io::stdin().read_line(&mut go).unwrap();
    let window = Duration::from_secs_f64(config.seconds);
    let start = Instant::now();
    let mut writes = 0u64;
    let mut slowest = Duration::ZERO;
    loop {
        let before = Instant::now();
        store.set(input, writes + 1).unwrap();
        let after = Instant::now();
        // A write that ends past the window does not count, so every
        // writer's count covers the same span and the counts add up.
        if after - start > window {
            break;
        }
        writes += 1;
        slowest = slowest.max(after - before);
    }
    println!("done {writes} {}", slowest.as_nanos());
}

fn bench_dir() -> tempfile::TempDir {
    let base = env::var_os("INCR_PERSIST_BENCH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/incr-persist-bench")
        });
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(base).unwrap()
}

fn run(writers: usize, config: &Config) {
    // Created up front so the writers do not race to create the schema.
    drop(open(config));
    let mut children: Vec<_> = (0..writers)
        .map(|id| {
            let mut child = Command::new(env::current_exe().unwrap())
                .args([
                    "0",
                    &config.seconds.to_string(),
                    durability_name(config.durability),
                ])
                .env(WRITER, id.to_string())
                .env("INCR_PERSIST_DB", &config.path)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            let lines = BufReader::new(child.stdout.take().unwrap()).lines();
            (child, lines)
        })
        .collect();
    for (_, lines) in &mut children {
        assert_eq!(lines.next().unwrap().unwrap(), "ready");
    }
    for (child, _) in &mut children {
        writeln!(child.stdin.as_mut().unwrap(), "go").unwrap();
    }
    let window = config.seconds;
    let mut total = 0u64;
    for (id, (mut child, mut lines)) in children.into_iter().enumerate() {
        let line = lines.next().unwrap().unwrap();
        let mut fields = line.split(' ').skip(1);
        let writes: u64 = fields.next().unwrap().parse().unwrap();
        let slowest = Duration::from_nanos(fields.next().unwrap().parse().unwrap());
        assert!(child.wait().unwrap().success());
        println!(
            "writer {id}: {writes} writes, {:.0} writes/s, slowest write {:.1} ms",
            writes as f64 / window,
            slowest.as_secs_f64() * 1e3
        );
        total += writes;
    }
    println!(
        "aggregate ({writers} writers, {}, {window}s): {:.0} writes/s, {:.1} us/write",
        durability_name(config.durability),
        total as f64 / window,
        window * 1e6 / total as f64
    );
}

fn durability_name(durability: SqliteDurability) -> &'static str {
    match durability {
        SqliteDurability::Full => "full",
        SqliteDurability::Buffered => "buffered",
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let writers: usize = args.first().map_or(2, |a| a.parse().unwrap());
    let seconds: f64 = args.get(1).map_or(10.0, |a| a.parse().unwrap());
    let durability = match args.get(2).map_or("full", String::as_str) {
        "full" => SqliteDurability::Full,
        "buffered" => SqliteDurability::Buffered,
        other => panic!("durability is full or buffered, not {other}"),
    };
    if let Ok(id) = env::var(WRITER) {
        let path = PathBuf::from(env::var_os("INCR_PERSIST_DB").unwrap());
        return writer(
            &id,
            &Config {
                path,
                seconds,
                durability,
            },
        );
    }
    let dir = bench_dir();
    let config = Config {
        path: dir.path().join("log.db"),
        seconds,
        durability,
    };
    run(writers, &config);
}
