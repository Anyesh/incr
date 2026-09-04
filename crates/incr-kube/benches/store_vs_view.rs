//! Published milestone benchmark: 50k Pods, a writer applying watch
//! events at the milestone's ~1,000 events/sec target, comparing
//! kube-rs's `Store::state_filter` against an `incr-kube` filtered
//! view's `values()` read.
//!
//! **Writer side, confirmed:** `Store::state_filter`'s own docs say the
//! store's read lock is held for the entire predicate scan;
//! `Writer::apply_watcher_event` takes that same lock to write
//! (`kube-runtime/src/reflector/store.rs`), so a write queues behind an
//! in-flight full-collection scan. `incr-kube`'s write path
//! (`KeyedCollection::upsert`) never touches the filtered view's own
//! lock, so it isn't subject to the same stall.
//!
//! **Reader side, more nuanced:** a filtered view's `values()` clones
//! its *current result set*, not the source collection, so it's O(view
//! size), not O(source size) the way `state_filter`'s scan is. At high
//! selectivity (most of the source matches) those are close to the same
//! number, which is exactly what this file's first scenario measures:
//! it shows `values()` costing *more* than `state_filter` per call under
//! concurrent readers, not less. Two separate costs stack there. First,
//! materializing a 25k-entry view has its own String-clone cost
//! (`Entry`'s key is an owned `ObjectRef`, not just an `Arc`); the
//! "reader (idle, no writer)" baseline below isolates this alone, since
//! the view is never dirty and no recompute ever runs. Second, under a
//! live writer that keeps the view dirty, `values()` holds the view's
//! output-collection read guard across that whole clone, so the
//! recompute that must take the matching write guard (to apply the new
//! delta) queues behind it; while that one reader holds the
//! wait-on-Computing claim, the other reader threads spin
//! (`incr-core/src/runtime.rs`'s `compute_one`) waiting on the same
//! node. That's the same shape as `Store::state_filter` (a read lock
//! held across a full scan blocking a writer), just relocated onto
//! `incr-kube`'s own reader path instead of kube-rs's writer path. The
//! second scenario, at low selectivity, is where the two really
//! diverge: `state_filter` still scans all 50k Pods to find the rare
//! matches, while `values()` only ever clones a small result set, so
//! neither cost above has much to bite on.
//!
//! Not a criterion harness: the milestone asks for reader p99 and
//! writer stall specifically, which criterion's default output doesn't
//! surface. This drives both systems from the same synthetic event
//! mix directly, times every individual read and write call, and
//! reports percentiles, plus the writer's actually achieved rate. The
//! writer paces against an absolute per-event deadline (not a
//! sleep-after-each-write interval), so a slow write eats into the
//! achieved rate exactly once instead of also losing the sleep's own
//! rounding-up overshoot on every following event; a real reflector
//! draining a backlogged watch stream behaves the same way.

use incr_concurrent::Runtime;
use incr_kube::{Entry, KeyedCollection};
use k8s_openapi::api::core::v1::{Pod, PodStatus};
use kube::runtime::reflector::{store, ObjectRef};
use kube::runtime::watcher;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const POD_COUNT: usize = 50_000;
const WRITE_EVENTS: usize = 5_000;
const TARGET_EVENTS_PER_SEC: u64 = 1_000;
const READER_THREADS: usize = 4;
const IDLE_BASELINE: Duration = Duration::from_millis(300);

struct Scenario {
    name: &'static str,
    /// Pod `idx` is Running unless `idx % pending_modulus == 0`; 2 gives
    /// a 50/50 split, 100 gives a rare 1-in-100 Pending cohort.
    pending_modulus: usize,
    /// The predicate the view/scan filters on. Paired with
    /// `pending_modulus` so each scenario reads whichever side is the
    /// rarer one, matching a real "find the pods not Running" query at
    /// low selectivity.
    predicate: fn(&Pod) -> bool,
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "balanced (50% match)",
        pending_modulus: 2,
        predicate: is_running,
    },
    Scenario {
        name: "sparse (1% match)",
        pending_modulus: 100,
        predicate: is_pending,
    },
];

fn phase_for(idx: usize, pending_modulus: usize) -> bool {
    !idx.is_multiple_of(pending_modulus)
}

fn make_pod(idx: usize, resource_version: &str, running: bool) -> Pod {
    let mut pod = Pod::default();
    pod.metadata.name = Some(format!("pod-{idx}"));
    pod.metadata.namespace = Some("default".to_string());
    pod.metadata.resource_version = Some(resource_version.to_string());
    pod.status = Some(PodStatus {
        phase: Some(if running { "Running" } else { "Pending" }.to_string()),
        ..Default::default()
    });
    pod
}

fn is_running(pod: &Pod) -> bool {
    pod.status
        .as_ref()
        .and_then(|s| s.phase.as_deref())
        .map(|phase| phase == "Running")
        .unwrap_or(false)
}

fn is_pending(pod: &Pod) -> bool {
    pod.status
        .as_ref()
        .and_then(|s| s.phase.as_deref())
        .map(|phase| phase == "Pending")
        .unwrap_or(false)
}

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * pct).round() as usize;
    sorted[idx]
}

fn report(name: &str, side: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let mean: Duration = samples.iter().sum::<Duration>() / samples.len() as u32;
    println!(
        "{name} {side}: n={} mean={:?} p50={:?} p99={:?} max={:?}",
        samples.len(),
        mean,
        percentile(&samples, 0.50),
        percentile(&samples, 0.99),
        samples.last().copied().unwrap_or_default(),
    );
}

/// Spawns `READER_THREADS` threads calling `read` in a tight loop,
/// timing every call, until `stop` is set. Returns one join handle per
/// thread; join them and flatten to get every sample.
fn spawn_readers<F>(stop: &Arc<AtomicBool>, read: F) -> Vec<JoinHandle<Vec<Duration>>>
where
    F: Fn() -> usize + Clone + Send + 'static,
{
    (0..READER_THREADS)
        .map(|_| {
            let stop = Arc::clone(stop);
            let read = read.clone();
            std::thread::spawn(move || {
                let mut local = Vec::new();
                while !stop.load(Ordering::Relaxed) {
                    let t0 = Instant::now();
                    let _matched = read();
                    local.push(t0.elapsed());
                }
                local
            })
        })
        .collect()
}

fn join_readers(handles: Vec<JoinHandle<Vec<Duration>>>) -> Vec<Duration> {
    handles
        .into_iter()
        .flat_map(|h| h.join().unwrap())
        .collect()
}

/// Runs `read` with no writer active for `IDLE_BASELINE`, isolating the
/// read path's own materialization cost from any writer-driven recompute
/// contention (see the module doc's "reader (idle, no writer)" line).
fn idle_reader_baseline<F>(name: &str, read: F)
where
    F: Fn() -> usize + Clone + Send + 'static,
{
    let stop = Arc::new(AtomicBool::new(false));
    let handles = spawn_readers(&stop, read);
    std::thread::sleep(IDLE_BASELINE);
    stop.store(true, Ordering::Relaxed);
    report(name, "reader (idle, no writer)", join_readers(handles));
}

/// Sleeps until `deadline` if it hasn't already passed. Pacing against
/// an absolute per-event deadline, not a fixed sleep after each write,
/// means a slow write costs exactly the time it took and nothing more;
/// a fixed post-write sleep would also add every sleep call's own
/// rounding-up overshoot on top, on every single event.
fn sleep_until(deadline: Instant) {
    if let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        std::thread::sleep(remaining);
    }
}

fn bench_kube_store(scenario: &Scenario) {
    let (reader, mut writer) = store::<Pod>();

    // Seed via a relist, the same way a real reflector populates a
    // fresh Store: Init, POD_COUNT InitApply, InitDone.
    writer.apply_watcher_event(&watcher::Event::Init);
    for idx in 0..POD_COUNT {
        let running = phase_for(idx, scenario.pending_modulus);
        writer.apply_watcher_event(&watcher::Event::InitApply(make_pod(idx, "1", running)));
    }
    writer.apply_watcher_event(&watcher::Event::InitDone);
    println!(
        "{}: seeded {} matches out of {POD_COUNT}",
        scenario.name,
        reader.state_filter(scenario.predicate).len(),
    );

    let label = format!("{} kube-rs Store::state_filter", scenario.name);
    idle_reader_baseline(&label, {
        let reader = reader.clone();
        let predicate = scenario.predicate;
        move || reader.state_filter(predicate).len()
    });

    let stop = Arc::new(AtomicBool::new(false));
    let readers = spawn_readers(&stop, {
        let reader = reader.clone();
        let predicate = scenario.predicate;
        move || reader.state_filter(predicate).len()
    });

    let mut writer_latencies = Vec::with_capacity(WRITE_EVENTS);
    let interval = Duration::from_micros(1_000_000 / TARGET_EVENTS_PER_SEC);
    let write_start = Instant::now();
    for i in 0..WRITE_EVENTS {
        let idx = i % POD_COUNT;
        let version = (i + 2).to_string();
        let pod = make_pod(idx, &version, phase_for(idx, scenario.pending_modulus));
        let t0 = Instant::now();
        writer.apply_watcher_event(&watcher::Event::Apply(pod));
        writer_latencies.push(t0.elapsed());
        sleep_until(write_start + interval * (i as u32 + 1));
    }
    let achieved_rate = WRITE_EVENTS as f64 / write_start.elapsed().as_secs_f64();

    stop.store(true, Ordering::Relaxed);
    let reader_latencies = join_readers(readers);

    println!(
        "{label}: writer achieved {achieved_rate:.0} events/sec (target {TARGET_EVENTS_PER_SEC})",
    );
    report(&label, "reader", reader_latencies);
    report(&label, "writer", writer_latencies);
}

fn bench_incr_kube_view(scenario: &Scenario) {
    let rt = Arc::new(Runtime::new());
    let keyed: Arc<KeyedCollection<ObjectRef<Pod>, Pod>> = Arc::new(KeyedCollection::new(&rt));

    for idx in 0..POD_COUNT {
        let running = phase_for(idx, scenario.pending_modulus);
        let pod = make_pod(idx, "1", running);
        keyed.upsert(&rt, ObjectRef::from(&pod), 1, pod);
    }

    let predicate = scenario.predicate;
    let view: incr_concurrent::IncrCollection<Entry<ObjectRef<Pod>, Pod>> = keyed
        .collection()
        .filter(&rt, move |e: &Entry<ObjectRef<Pod>, Pod>| {
            predicate(e.value())
        });
    println!(
        "{}: seeded {} matches out of {POD_COUNT}",
        scenario.name,
        view.values(&rt).len(),
    );

    let label = format!("{} incr-kube filtered view", scenario.name);
    idle_reader_baseline(&label, {
        let rt = Arc::clone(&rt);
        let view = view.clone();
        move || view.values(&rt).len()
    });

    let stop = Arc::new(AtomicBool::new(false));
    let readers = spawn_readers(&stop, {
        let rt = Arc::clone(&rt);
        let view = view.clone();
        move || view.values(&rt).len()
    });

    let mut writer_latencies = Vec::with_capacity(WRITE_EVENTS);
    let interval = Duration::from_micros(1_000_000 / TARGET_EVENTS_PER_SEC);
    let write_start = Instant::now();
    for i in 0..WRITE_EVENTS {
        let idx = i % POD_COUNT;
        let version = i as u64 + 2;
        let running = phase_for(idx, scenario.pending_modulus);
        let pod = make_pod(idx, &version.to_string(), running);
        let key = ObjectRef::from(&pod);
        let t0 = Instant::now();
        keyed.upsert(&rt, key, version, pod);
        writer_latencies.push(t0.elapsed());
        sleep_until(write_start + interval * (i as u32 + 1));
    }
    let achieved_rate = WRITE_EVENTS as f64 / write_start.elapsed().as_secs_f64();

    stop.store(true, Ordering::Relaxed);
    let reader_latencies = join_readers(readers);

    println!(
        "{label}: writer achieved {achieved_rate:.0} events/sec (target {TARGET_EVENTS_PER_SEC})",
    );
    report(&label, "reader", reader_latencies);
    report(&label, "writer", writer_latencies);
}

fn main() {
    for scenario in SCENARIOS {
        bench_kube_store(scenario);
        bench_incr_kube_view(scenario);
    }
}
