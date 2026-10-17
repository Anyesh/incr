//! Runs a test body in a child process so a hang can be observed and killed.
//!
//! Included by path from the lib's `cfg(test)` build and from integration
//! tests (`#[path = "common/child.rs"] mod child;`), because a lib-level
//! `cfg(test)` module is invisible to `tests/`.
#![allow(dead_code)]

use std::env;
use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const CHILD: &str = "INCR_TEST_CHILD";

pub const HANG_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug)]
pub enum Outcome {
    Clean,
    Panicked { output: String },
    Killed,
}

impl Outcome {
    pub fn assert_clean(&self) {
        assert!(
            matches!(self, Outcome::Clean),
            "expected a clean exit, got {self:?}"
        );
    }

    pub fn assert_killed(&self) {
        assert!(
            matches!(self, Outcome::Killed),
            "expected a hang killed by the timeout, got {self:?}"
        );
    }

    pub fn assert_panicked_with(&self, message: &str) {
        match self {
            Outcome::Panicked { output } if output.contains(message) => {}
            other => panic!("expected a panic containing {message:?}, got {other:?}"),
        }
    }
}

/// Runs `body` in a re-executed copy of this test binary with the 10 s timeout.
///
/// `test` is the full test path as libtest names it (matched with `--exact`).
/// In the child the body runs and the process exits; in the parent the child's
/// fate is returned. Callers must add `#[cfg_attr(miri, ignore)]` because Miri
/// cannot spawn processes.
pub fn run_in_child(test: &str, body: impl FnOnce()) -> Outcome {
    run_in_child_within(test, HANG_TIMEOUT, body)
}

pub fn run_in_child_within(test: &str, timeout: Duration, body: impl FnOnce()) -> Outcome {
    if env::var_os(CHILD).is_some() {
        body();
        std::process::exit(0);
    }

    let mut child = Command::new(env::current_exe().unwrap())
        .args([test, "--exact", "--test-threads=1", "--nocapture"])
        .env(CHILD, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let drain = |mut pipe: Box<dyn Read + Send>| {
        thread::spawn(move || {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text);
            text
        })
    };
    let stdout = drain(Box::new(child.stdout.take().unwrap()));
    let stderr = drain(Box::new(child.stderr.take().unwrap()));

    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            break None;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let output = format!("{}{}", stdout.join().unwrap(), stderr.join().unwrap());
    match status {
        None => Outcome::Killed,
        Some(status) if status.success() => Outcome::Clean,
        Some(_) => Outcome::Panicked { output },
    }
}
