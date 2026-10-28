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

/// How many times longer a child gets in a sanitized build
/// (`INCR_TEST_REDUCED`, which the TSan job sets), where it runs several
/// times slower and a slow runner stretches that again.
const SANITIZED_SLOWDOWN: u32 = 6;

fn scaled(timeout: Duration) -> Duration {
    if env::var_os("INCR_TEST_REDUCED").is_some() {
        timeout * SANITIZED_SLOWDOWN
    } else {
        timeout
    }
}

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

/// Runs `body` in a re-executed copy of this test binary with [`HANG_TIMEOUT`].
///
/// `test` is the full test path as libtest names it (matched with `--exact`).
/// In the child the body runs and the process exits; in the parent the child's
/// fate is returned. Callers must add `#[cfg_attr(miri, ignore)]` because Miri
/// cannot spawn processes.
pub fn run_in_child(test: &str, body: impl FnOnce()) -> Outcome {
    run_in_child_within(test, HANG_TIMEOUT, body)
}

/// Runs `body` with no other test thread in the process: in a child, or
/// directly under Miri, whose test runs are single-threaded. For a body
/// that asserts what one reclaim of haphazard's global domain drops, which
/// a reclaim pass or a hazard on another test thread can hold back.
pub fn run_alone(test: &str, body: impl FnOnce()) {
    if cfg!(miri) {
        body();
    } else {
        run_in_child(test, body).assert_clean();
    }
}

/// [`run_in_child`] with `timeout` for the plain build, scaled up in a
/// sanitized build like [`HANG_TIMEOUT`].
pub fn run_in_child_within(test: &str, timeout: Duration, body: impl FnOnce()) -> Outcome {
    if env::var_os(CHILD).is_some() {
        body();
        std::process::exit(0);
    }

    let mut child = Command::new(env::current_exe().unwrap())
        .args([
            test,
            "--exact",
            "--include-ignored",
            "--test-threads=1",
            "--nocapture",
        ])
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

    let deadline = Instant::now() + scaled(timeout);
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
        Some(status) if status.success() => {
            // The body exits before libtest's summary, so a summary means
            // the body never ran: `test` named no test.
            assert!(
                !output.contains("test result:"),
                "the child never ran a test named {test}: {output}"
            );
            Outcome::Clean
        }
        Some(_) => Outcome::Panicked { output },
    }
}
