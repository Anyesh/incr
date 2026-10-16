//! Kills a `churn` writer with SIGKILL while it appends, over and over on
//! one log, and checks after each kill that a restarted writer and a
//! read-only process both see every acknowledged op, at most the one op
//! that was in flight, and the revenue the writer reported.
//!
//! The kill lands mid-write without any sleeping: the writer never waits
//! for the parent, so by the time the parent has read the ack it was
//! waiting for, the writer is already inside its next op.
//!
//! SIGKILL does not discard data a process already handed to `write(2)`,
//! so this proves restart correctness (recovery, replay, torn-tail
//! handling, and compactions cut short), not durability against power loss.
#![cfg(unix)]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::{Command, Stdio};

use incr_persist_config::{churn_keys, churn_op, Op, Order, View};

const BIN: &str = env!("CARGO_BIN_EXE_incr-persist-config");
const ITERATIONS: usize = 16;
const MAX_ACKS_BEFORE_KILL: u64 = 40;
const SIGKILL: i32 = 9;

/// The shop's state computed in plain Rust, independently of incr, as the
/// oracle for both the log's contents and the revenue query.
#[derive(Clone)]
struct Model {
    config: BTreeMap<String, u64>,
    orders: BTreeMap<Order, usize>,
}

impl Model {
    fn new() -> Model {
        Model {
            config: churn_keys().into_iter().map(|key| (key, 0)).collect(),
            orders: BTreeMap::new(),
        }
    }

    fn apply(&mut self, op: &Op) {
        match op {
            Op::Set { key, value } => {
                self.config.insert(key.clone(), *value);
            }
            Op::Order(order) => *self.orders.entry(order.clone()).or_default() += 1,
            Op::Cancel(order) => {
                let count = self.orders.get_mut(order).expect("cancels a live order");
                *count -= 1;
                if *count == 0 {
                    self.orders.remove(order);
                }
            }
            Op::PriceAndOrder { key, value, order } => {
                self.config.insert(key.clone(), *value);
                *self.orders.entry(order.clone()).or_default() += 1;
            }
        }
    }

    fn view(&self) -> View {
        let orders: Vec<Order> = self
            .orders
            .iter()
            .flat_map(|(order, &count)| std::iter::repeat_n(order.clone(), count))
            .collect();
        let gross: u64 = orders
            .iter()
            .map(|o| o.qty * self.config.get(&format!("price.{}", o.sku)).unwrap_or(&0))
            .sum();
        let discount = self
            .config
            .get("discount_pct")
            .copied()
            .unwrap_or(0)
            .min(100);
        View {
            config: self.config.clone(),
            orders,
            revenue: gross * (100 - discount) / 100,
        }
    }
}

fn show(dir: &Path, read_only: bool) -> String {
    let mut command = Command::new(BIN);
    command.arg(dir).arg("show");
    if read_only {
        command.arg("--read-only");
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "show (read_only={read_only}) failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn parse_ack(line: &str) -> (u64, u64) {
    let parsed = line.strip_prefix("ack ").and_then(|rest| {
        let (i, revenue) = rest.split_once(" revenue=")?;
        Some((i.parse().ok()?, revenue.parse().ok()?))
    });
    parsed.unwrap_or_else(|| panic!("unexpected line from the writer: {line:?}"))
}

/// A fixed xorshift sequence, so the ack counts the kills wait for vary
/// widely but the same way on every run.
fn kill_points() -> impl Iterator<Item = u64> {
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    std::iter::from_fn(move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        Some(x % (MAX_ACKS_BEFORE_KILL + 1))
    })
}

#[test]
fn acknowledged_writes_survive_sigkill_mid_write() {
    let dir = tempfile::tempdir().unwrap();
    let mut landed = 0u64;
    let mut model = Model::new();
    let mut in_flight_landed = 0;

    for (iteration, wait_for) in kill_points().take(ITERATIONS).enumerate() {
        // The first writer appends the declarations before it says ready,
        // so from then on every kill falls inside an op the model knows.
        let wait_for = if iteration == 0 {
            wait_for.max(1)
        } else {
            wait_for
        };
        let mut writer = Command::new(BIN)
            .arg(dir.path())
            .args(["churn", &landed.to_string()])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut lines = BufReader::new(writer.stdout.take().unwrap()).lines();
        assert_eq!(lines.next().unwrap().unwrap(), "ready");
        let mut acks = Vec::new();
        while (acks.len() as u64) < wait_for {
            acks.push(parse_ack(&lines.next().unwrap().unwrap()));
        }

        writer.kill().unwrap();
        let status = writer.wait().unwrap();
        assert_eq!(
            status.signal(),
            Some(SIGKILL),
            "iteration {iteration}: the writer stopped on its own ({status}) instead of being killed"
        );
        // Acks the writer printed after the one the parent waited for are
        // still in the pipe; they were acknowledged before the kill.
        for line in lines {
            acks.push(parse_ack(&line.unwrap()));
        }

        let mut expected = model.clone();
        for (offset, &(i, revenue)) in acks.iter().enumerate() {
            assert_eq!(i, landed + offset as u64, "acks arrive in op order");
            expected.apply(&churn_op(i));
            assert_eq!(
                revenue,
                expected.view().revenue,
                "iteration {iteration}: the writer's live query after op {i}"
            );
        }
        let acked = landed + acks.len() as u64;

        let read_only_before_recovery = show(dir.path(), true);
        let restarted = show(dir.path(), false);
        let read_only_after_recovery = show(dir.path(), true);

        let mut with_in_flight = expected.clone();
        with_in_flight.apply(&churn_op(acked));
        let (now_landed, now_model) = if restarted == expected.view().to_string() {
            (acked, expected)
        } else if restarted == with_in_flight.view().to_string() {
            in_flight_landed += 1;
            (acked + 1, with_in_flight)
        } else {
            panic!(
                "iteration {iteration}: after {acked} acknowledged ops the restarted log shows\n\
                 {restarted}\nwhich is neither the acknowledged state\n{}\nnor that plus op {acked}\n{}",
                expected.view(),
                with_in_flight.view()
            );
        };
        assert_eq!(
            read_only_before_recovery, restarted,
            "iteration {iteration}: a read-only process before the writer recovered the log"
        );
        assert_eq!(
            read_only_after_recovery, restarted,
            "iteration {iteration}: a read-only process after the writer recovered the log"
        );
        landed = now_landed;
        model = now_model;
    }

    assert!(landed > 0);
    eprintln!(
        "{ITERATIONS} kills, {landed} ops in the log, the in-flight op had landed in {in_flight_landed} of them"
    );
}
