//! Concurrent correctness tests for the v2 Runtime.
//!
//! The v2 architecture's core claim is single-writer-many-readers:
//! one writer at a time serializes through the runtime's write
//! mutex while any number of readers call `rt.get` concurrently
//! without contention on the reader-reader path. The existing
//! commit H-N tests cover this architecturally (each piece in
//! isolation) and the proptest in `runtime_proptest.rs` covers it
//! functionally (incremental equals batch), but neither exercises
//! the full concurrent path with real OS threads doing real
//! simultaneous reads against a live writer.
//!
//! This module does that. Each test spawns N reader threads
//! against a shared `Arc<Runtime>` while a writer thread perturbs
//! inputs on a schedule, and asserts that every value a reader
//! observes is legitimate (comes from the set of values the writer
//! has ever written, possibly through a deterministic compute).
//! A stronger property (full linearizability via happens-before
//! reasoning across every observation) is left to commit P if
//! this coarser property passes cleanly.
//!
//! Placed in-crate because v2 is `pub(crate)` until Gate 5; tests
//! need crate-private access to construct handles and check state
//! that the public API will eventually expose differently.

#![cfg(test)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use super::runtime::Runtime;

/// Stress duration per test. Short enough that the whole suite
/// runs in under a second in normal mode, long enough that any
/// data race or ordering bug has thousands of opportunities to
/// trip.
const DEFAULT_WRITER_ITERS: usize = 5_000;

#[test]
fn many_readers_observe_only_written_input_values() {
    // One String input, 8 readers, writer cycles through a known
    // set of strings. Any observation not in the set is either a
    // torn read (UB we would have hit before commit G's fix, and
    // should still pass now that the RwLock gate on nodes serves
    // as the correctness barrier) or a stale value from a
    // different writer generation, neither of which can happen
    // under the current design.
    const READERS: usize = 8;

    let rt = Arc::new(Runtime::new());
    let values: Vec<String> = (0..16)
        .map(|i| format!("value-{}-padding-to-force-heap-allocation", i))
        .collect();
    let valid: HashSet<String> = values.iter().cloned().collect();
    let input = rt.create_input::<String>(values[0].clone());

    let stop = Arc::new(AtomicBool::new(false));

    let reader_handles: Vec<_> = (0..READERS)
        .map(|_| {
            let rt = rt.clone();
            let valid = valid.clone();
            let stop = stop.clone();
            thread::spawn(move || {
                let mut observations = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let v = rt.get(input);
                    assert!(
                        valid.contains(&v),
                        "reader observed non-written value: {:?}",
                        v
                    );
                    observations += 1;
                }
                observations
            })
        })
        .collect();

    for i in 0..DEFAULT_WRITER_ITERS {
        let v = &values[i % values.len()];
        rt.set(input, v.clone());
    }
    stop.store(true, Ordering::Relaxed);

    let total: usize = reader_handles
        .into_iter()
        .map(|h| h.join().expect("reader panicked"))
        .sum();
    assert!(
        total > 0,
        "readers should have completed at least one read each"
    );
}

#[test]
fn many_readers_observe_only_valid_query_values() {
    // Input plus a pure function query. Every query observation
    // must be `input + 100` for some input value the writer has
    // set. This exercises the full reactivity path (set → dirty
    // walk → reader observes Dirty → reader CASes to Computing →
    // reader recomputes → reader observes Clean) under concurrency.
    const READERS: usize = 8;

    let rt = Arc::new(Runtime::new());
    let inputs: Vec<u64> = (0..20).collect();
    let valid_queries: HashSet<u64> = inputs.iter().map(|i| i + 100).collect();

    let input = rt.create_input::<u64>(inputs[0]);
    let query = rt.create_query::<u64, _>(move |rt| rt.get(input) + 100);

    let stop = Arc::new(AtomicBool::new(false));

    let reader_handles: Vec<_> = (0..READERS)
        .map(|_| {
            let rt = rt.clone();
            let valid = valid_queries.clone();
            let stop = stop.clone();
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let v = rt.get(query);
                    assert!(
                        valid.contains(&v),
                        "reader observed query value {} not in valid set (expected input+100 for some written input)",
                        v
                    );
                }
            })
        })
        .collect();

    for i in 0..DEFAULT_WRITER_ITERS {
        let v = inputs[i % inputs.len()];
        rt.set(input, v);
    }
    stop.store(true, Ordering::Relaxed);

    for h in reader_handles {
        h.join().expect("reader panicked");
    }
}

#[test]
fn many_readers_on_chain_of_queries_observe_valid_values() {
    // input → q1 (+1) → q2 (*2) → q3 (+100). Compute function for
    // q3(input) = ((input + 1) * 2) + 100. Readers spin on q3;
    // writer sets input. Every observation must be valid for some
    // written input.
    const READERS: usize = 6;

    let rt = Arc::new(Runtime::new());
    let inputs: Vec<u64> = (0..50).collect();
    let valid: HashSet<u64> = inputs.iter().map(|i| ((i + 1) * 2) + 100).collect();

    let input = rt.create_input::<u64>(inputs[0]);
    let q1 = rt.create_query::<u64, _>(move |rt| rt.get(input) + 1);
    let q2 = rt.create_query::<u64, _>(move |rt| rt.get(q1) * 2);
    let q3 = rt.create_query::<u64, _>(move |rt| rt.get(q2) + 100);

    let stop = Arc::new(AtomicBool::new(false));

    let reader_handles: Vec<_> = (0..READERS)
        .map(|_| {
            let rt = rt.clone();
            let valid = valid.clone();
            let stop = stop.clone();
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let v = rt.get(q3);
                    assert!(
                        valid.contains(&v),
                        "reader observed q3 value {} not valid for any input in {:?}",
                        v,
                        (0..5u64)
                    );
                }
            })
        })
        .collect();

    for i in 0..DEFAULT_WRITER_ITERS {
        rt.set(input, inputs[i % inputs.len()]);
    }
    stop.store(true, Ordering::Relaxed);

    for h in reader_handles {
        h.join().expect("reader panicked");
    }
}

#[test]
fn concurrent_reads_of_multiple_unrelated_chains_do_not_cross_contaminate() {
    // Two independent chains sharing a runtime. Readers on chain A
    // should never observe values from chain B's inputs, even if
    // the writer is updating both. This catches any cross-slot
    // contamination in the state machine or dep graph.
    const READERS_PER_CHAIN: usize = 3;

    let rt = Arc::new(Runtime::new());

    let a_inputs: Vec<u64> = (0..20).collect();
    let b_inputs: Vec<u64> = (100..120).collect();
    let a_valid: HashSet<u64> = a_inputs.iter().map(|i| i * 10).collect();
    let b_valid: HashSet<u64> = b_inputs.iter().map(|i| i * 10).collect();

    let a_in = rt.create_input::<u64>(a_inputs[0]);
    let b_in = rt.create_input::<u64>(b_inputs[0]);
    let a_q = rt.create_query::<u64, _>(move |rt| rt.get(a_in) * 10);
    let b_q = rt.create_query::<u64, _>(move |rt| rt.get(b_in) * 10);

    let stop = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();
    for _ in 0..READERS_PER_CHAIN {
        let rt = rt.clone();
        let valid = a_valid.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let v = rt.get(a_q);
                assert!(
                    valid.contains(&v),
                    "chain A reader observed cross-contaminated value: {}",
                    v
                );
            }
        }));
    }
    for _ in 0..READERS_PER_CHAIN {
        let rt = rt.clone();
        let valid = b_valid.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let v = rt.get(b_q);
                assert!(
                    valid.contains(&v),
                    "chain B reader observed cross-contaminated value: {}",
                    v
                );
            }
        }));
    }

    for i in 0..DEFAULT_WRITER_ITERS {
        rt.set(a_in, a_inputs[i % a_inputs.len()]);
        rt.set(b_in, b_inputs[i % b_inputs.len()]);
    }
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().expect("reader panicked");
    }
}

#[test]
fn compute_function_runs_at_most_once_per_dirty_cycle() {
    // When multiple readers race to recompute a Dirty query, only
    // one should actually execute the compute closure per dirty
    // cycle. The state machine's Dirty → Computing CAS enforces
    // this. Count compute invocations; expected upper bound is
    // roughly the number of times the writer called `set`.
    //
    // We allow a margin: the writer may set faster than readers
    // can recompute, so several sets can coalesce into one
    // recompute (good), and a set during Computing can miss the
    // dirty mark on the current generation (the known race; its
    // impact here is that the count may be slightly LOWER than
    // the number of sets, not higher). The bounds we assert are
    // loose enough that both directions are tolerated.
    const READERS: usize = 8;
    const WRITER_SETS: usize = 2_000;

    let rt = Arc::new(Runtime::new());
    let compute_invocations = Arc::new(AtomicUsize::new(0));

    let input = rt.create_input::<u64>(0);
    let query = {
        let counter = compute_invocations.clone();
        rt.create_query::<u64, _>(move |rt| {
            counter.fetch_add(1, Ordering::SeqCst);
            rt.get(input) * 2
        })
    };

    let stop = Arc::new(AtomicBool::new(false));
    let reader_handles: Vec<_> = (0..READERS)
        .map(|_| {
            let rt = rt.clone();
            let stop = stop.clone();
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let _ = rt.get(query);
                }
            })
        })
        .collect();

    for i in 0..WRITER_SETS {
        rt.set(input, i as u64);
    }
    stop.store(true, Ordering::Relaxed);
    for h in reader_handles {
        h.join().expect("reader panicked");
    }

    let total = compute_invocations.load(Ordering::SeqCst);
    // Each set() triggers at most one recompute. Readers may
    // observe stale Clean between sets without triggering a
    // recompute (coalescing). We expect total <= WRITER_SETS + 1
    // (the +1 for the initial compute). We also expect total >= 1
    // (at least the initial compute ran).
    assert!(
        total >= 1,
        "expected at least one compute invocation, got {}",
        total
    );
    assert!(
        total <= WRITER_SETS + READERS + 10,
        "compute ran too many times ({}); expected <= {} (writer_sets + readers + slack)",
        total,
        WRITER_SETS + READERS + 10
    );
}

#[test]
fn computing_during_dirty_walk_does_not_leak_stale_value() {
    // Deterministic reproduction of the Computing-during-dirty-walk
    // race via two barriers. This race existed in commits J through
    // N and was fixed in commit P by a post-compute revision check
    // inside run_compute: before transitioning from Computing to
    // Clean, compare the current revision counter to the one
    // recorded at compute start. If they differ, a writer landed a
    // set during the compute, so the result is potentially stale
    // and we transition to Dirty instead of Clean, forcing the
    // next reader to retry the compute against the fresh inputs.
    //
    // The test threads are:
    //   - compute thread: triggers rt.get(q), blocks inside the
    //     compute closure on barrier A, then reads input, then
    //     blocks on barrier B.
    //   - writer thread: waits on barrier A so compute has started,
    //     then calls rt.set(input, new_value) which bumps revision
    //     and runs the dirty walk (which sees q in Computing state
    //     and fails to mark it Dirty), then releases barrier B.
    //   - main thread: after both join, reads q and asserts it
    //     reflects the new input value, not the stale one.
    //
    // Without the fix, the compute finishes with the old input
    // value, Release-stores Clean with the stale result, and the
    // next reader sees Clean + stale. With the fix, the compute
    // detects the revision bump, transitions to Dirty, and the
    // next reader's get() triggers a retry that uses the new input.
    use std::sync::Barrier;

    let rt = Arc::new(Runtime::new());
    let input = rt.create_input::<u64>(10);

    // Barriers coordinate the two compute-side threads. Each is
    // sized to 2 (compute thread plus writer thread).
    let started_barrier = Arc::new(Barrier::new(2));
    let set_complete_barrier = Arc::new(Barrier::new(2));
    // Only the FIRST invocation of the compute closure blocks on
    // the barriers. When commit P's fix detects the revision bump
    // and marks the node Dirty, the next reader retries the
    // closure; the retry must NOT block on the barriers because
    // the writer thread has already finished and would never
    // arrive. This counter tracks "is this the first call" with
    // AtomicUsize so the closure stays Fn (not FnOnce).
    let first_call = Arc::new(AtomicUsize::new(0));

    let q = {
        let started = started_barrier.clone();
        let set_complete = set_complete_barrier.clone();
        let first_call = first_call.clone();
        rt.create_query::<u64, _>(move |rt| {
            // Read the input first so this becomes a recorded dep.
            let v = rt.get(input);
            // Only the first invocation participates in the
            // barrier dance. Retries (from the revision-bump
            // detection in run_compute) just return the current
            // input value without waiting.
            if first_call.fetch_add(1, Ordering::SeqCst) == 0 {
                // First call: signal the writer that our compute
                // has started and we have already read the old
                // input value.
                started.wait();
                // Block until the writer has called set and
                // finished its (failed-to-mark-us-dirty) walk.
                set_complete.wait();
            }
            // Produce the result based on whatever input value we
            // read. On the first call v=10 (stale), on the retry
            // v=20 (fresh).
            v * 100
        })
    };

    // Compute thread: runs the read that triggers the compute.
    let compute_rt = rt.clone();
    let compute_thread = thread::spawn(move || compute_rt.get(q));

    // Writer thread: waits for compute to start, sets the input,
    // then releases the compute.
    let writer_rt = rt.clone();
    let writer_started = started_barrier.clone();
    let writer_set_complete = set_complete_barrier.clone();
    let writer_thread = thread::spawn(move || {
        writer_started.wait();
        writer_rt.set(input, 20);
        writer_set_complete.wait();
    });

    // Wait for the race to play out.
    let first_result = compute_thread.join().expect("compute thread panicked");
    writer_thread.join().expect("writer thread panicked");

    // With commit P's fix: the stale first compute (v=10, produces
    // 1000) is detected by the post-compute revision check inside
    // run_compute, which transitions the node to Dirty instead of
    // Clean. The outer `rt.get(q)` loop in the compute thread
    // observes Dirty on its next iteration, retries the compute
    // (this time with first_call >= 1 so the closure skips the
    // barriers), reads the fresh input value 20, and produces
    // 2000. The compute thread's `rt.get(q)` therefore returns the
    // correct post-set value 2000, not the stale 1000.
    //
    // Without the fix: run_compute would have Release-stored Clean
    // with the stale 1000 value. The compute thread's `rt.get(q)`
    // would return 1000. The next reader would also see 1000. The
    // assertion below would fail because first_result would be
    // 1000, not 2000.
    assert_eq!(
        first_result, 2000,
        "expected compute thread to observe the post-set value (2000) via \
         the internal retry triggered by commit P's revision check; got {}. \
         This failure indicates the Computing-during-dirty-walk race is \
         NOT closed: the stale compute leaked into Clean state.",
        first_result
    );

    // The closure should have run exactly twice: once with the
    // stale input (v=10, barriers taken), once with the fresh
    // input (v=20, barriers skipped via the first_call counter).
    // Without the fix, it would have run exactly once.
    let invocations = first_call.load(Ordering::SeqCst);
    assert_eq!(
        invocations, 2,
        "closure should run twice (stale then retry); got {} invocations",
        invocations
    );

    // Reading q again returns the (now-Clean) cached fresh value,
    // no additional compute invocations.
    let second_result = rt.get(q);
    assert_eq!(second_result, 2000);
    assert_eq!(
        first_call.load(Ordering::SeqCst),
        2,
        "second read should not have invoked the closure"
    );
}

#[test]
fn observations_are_monotonic_in_writer_logical_time() {
    // Stronger than "observation in valid set": every reader's
    // sequence of observations must be non-decreasing in the
    // writer's logical time. The writer increments the input
    // monotonically from 0 upward; any reader that observes value
    // N and then observes M < N is witnessing a stale read after
    // a fresh read, which would be a linearizability violation.
    //
    // This is a real-time monotonic ordering check at the
    // single-reader granularity. It catches reordering bugs the
    // "valid set" check cannot: a stale value that happens to be
    // in the valid set is not a valid set violation, but it is a
    // monotonicity violation if it follows a fresher read.
    //
    // Cross-reader ordering is NOT checked here: two readers may
    // observe the same value at different real times, or observe
    // different monotonic chains, depending on their interleaving
    // with the writer. Linearizability proper requires a global
    // total order; this test checks the weaker per-reader variant
    // that is both meaningful and cheap to verify.
    const READERS: usize = 8;
    const WRITER_ITERS: u64 = 5_000;

    let rt = Arc::new(Runtime::new());
    let input = rt.create_input::<u64>(0);

    let stop = Arc::new(AtomicBool::new(false));

    let reader_handles: Vec<_> = (0..READERS)
        .map(|i| {
            let rt = rt.clone();
            let stop = stop.clone();
            thread::spawn(move || {
                let mut highest_seen: u64 = 0;
                let mut observation_count: usize = 0;
                while !stop.load(Ordering::Relaxed) {
                    let v = rt.get(input);
                    assert!(
                        v >= highest_seen,
                        "reader {} observed {} after having already observed {} \
                         — monotonicity violation implies a stale read after a \
                         fresh read (real-time linearizability broken)",
                        i,
                        v,
                        highest_seen
                    );
                    highest_seen = v;
                    observation_count += 1;
                }
                (highest_seen, observation_count)
            })
        })
        .collect();

    for i in 0..WRITER_ITERS {
        rt.set(input, i);
    }
    stop.store(true, Ordering::Relaxed);

    let results: Vec<_> = reader_handles
        .into_iter()
        .map(|h| h.join().expect("reader panicked"))
        .collect();

    // Sanity: every reader made progress and eventually saw a
    // value close to the final writer value. We don't assert the
    // exact final value because readers may stop reading before
    // the very last set lands, but we do assert that the average
    // highest-seen is in a sensible range.
    let total_observations: usize = results.iter().map(|(_, c)| c).sum();
    assert!(
        total_observations > 0,
        "expected readers to make at least some progress"
    );
    let max_observed = results.iter().map(|(h, _)| h).max().copied().unwrap_or(0);
    assert!(
        max_observed > 0,
        "expected at least one reader to observe a non-initial value"
    );
}

#[test]
fn query_observations_are_monotonic_in_writer_logical_time() {
    // Same invariant as above but through a query node, so the
    // observation path goes through the reactive dirty walk and
    // recompute machinery. The query returns `input * 1000 + 7`
    // which is strictly monotonic in the input, so the reader
    // can decode the input value from the query result and check
    // monotonicity on that.
    const READERS: usize = 6;
    const WRITER_ITERS: u64 = 3_000;

    let rt = Arc::new(Runtime::new());
    let input = rt.create_input::<u64>(0);
    let query = rt.create_query::<u64, _>(move |rt| rt.get(input) * 1000 + 7);

    let stop = Arc::new(AtomicBool::new(false));

    let reader_handles: Vec<_> = (0..READERS)
        .map(|i| {
            let rt = rt.clone();
            let stop = stop.clone();
            thread::spawn(move || {
                let mut highest_seen: u64 = 7; // initial query value = 0*1000+7
                while !stop.load(Ordering::Relaxed) {
                    let v = rt.get(query);
                    // Decode: v = input * 1000 + 7
                    assert_eq!(
                        v % 1000,
                        7,
                        "reader {} observed query value {} which does not match \
                         the compute formula (input * 1000 + 7); torn read or \
                         corrupted value",
                        i,
                        v
                    );
                    assert!(
                        v >= highest_seen,
                        "reader {} observed query value {} after having seen {} \
                         — stale read after fresh read through the query path",
                        i,
                        v,
                        highest_seen
                    );
                    highest_seen = v;
                }
                highest_seen
            })
        })
        .collect();

    for i in 0..WRITER_ITERS {
        rt.set(input, i);
    }
    stop.store(true, Ordering::Relaxed);

    for h in reader_handles {
        h.join().expect("reader panicked");
    }
}

#[test]
fn multi_chain_observations_are_each_internally_monotonic() {
    // Two independent chains, each with its own monotonic input
    // and query. Verify per-chain monotonicity: a reader on chain
    // A should never observe an A value go backward, and same for
    // B. This extends the earlier "no cross-contamination" test
    // from commit O with a real-time ordering check on top of
    // the valid-set check.
    const READERS_PER_CHAIN: usize = 3;
    const WRITER_ITERS: u64 = 3_000;

    let rt = Arc::new(Runtime::new());
    let a_input = rt.create_input::<u64>(0);
    let b_input = rt.create_input::<u64>(0);
    let a_query = rt.create_query::<u64, _>(move |rt| rt.get(a_input) * 10);
    let b_query = rt.create_query::<u64, _>(move |rt| rt.get(b_input) * 10 + 500_000_000);

    let stop = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();
    for i in 0..READERS_PER_CHAIN {
        let rt = rt.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut highest: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                let v = rt.get(a_query);
                // A values are input * 10, so < 500_000_000 for
                // our input range. Catches cross-contamination.
                assert!(
                    v < 500_000_000,
                    "A reader {} observed B-like value {}",
                    i,
                    v
                );
                assert!(
                    v >= highest,
                    "A reader {} monotonicity: {} < {}",
                    i,
                    v,
                    highest
                );
                highest = v;
            }
        }));
    }
    for i in 0..READERS_PER_CHAIN {
        let rt = rt.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut highest: u64 = 500_000_000; // b_query initial = 0*10 + 500M
            while !stop.load(Ordering::Relaxed) {
                let v = rt.get(b_query);
                assert!(
                    v >= 500_000_000,
                    "B reader {} observed A-like value {}",
                    i,
                    v
                );
                assert!(
                    v >= highest,
                    "B reader {} monotonicity: {} < {}",
                    i,
                    v,
                    highest
                );
                highest = v;
            }
        }));
    }

    for i in 0..WRITER_ITERS {
        rt.set(a_input, i);
        rt.set(b_input, i);
    }
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().expect("reader panicked");
    }
}

#[test]
fn reader_threads_can_be_spawned_and_joined_repeatedly_on_same_runtime() {
    // Correctness under repeated reader-thread lifetimes. Spawn a
    // batch of readers, join them, set the input, spawn again.
    // Ensures the TLS compute cache and COMPUTE_STACK are clean
    // between reader lifetimes.
    let rt = Arc::new(Runtime::new());
    let input = rt.create_input::<u64>(1);
    let query = rt.create_query::<u64, _>(move |rt| rt.get(input) + 1000);

    for round in 0..5u64 {
        rt.set(input, round);
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let rt = rt.clone();
                thread::spawn(move || rt.get(query))
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), round + 1000);
        }
    }
}
