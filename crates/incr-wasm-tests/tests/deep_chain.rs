#![cfg(target_arch = "wasm32")]

use std::sync::atomic::{AtomicU32, Ordering::SeqCst};
use std::sync::{Arc, OnceLock};

use incr_core::{CycleStrategy, Incr, Local, QueryOpts, Replay, Runtime, DEFAULT_MAX_FRAMES};
use wasm_bindgen_test::wasm_bindgen_test;

/// A first compute of a 100k chain of replayable queries at the default
/// budget nests at most 256 frames, inside the harness's shadow stack and
/// V8 limit, and runs each node at most twice.
#[wasm_bindgen_test]
fn chain_of_100k_replays_within_the_frame_budget() {
    const N: usize = 100_000;
    let rt: Runtime<Local> = Runtime::new();
    rt.set_placeholder(0_u64);
    rt.set_max_frames(DEFAULT_MAX_FRAMES);
    let input = rt.create_input(0_u64);
    let runs: Arc<Vec<AtomicU32>> = Arc::new((0..N).map(|_| AtomicU32::new(0)).collect());
    let mut top = input;
    for i in (0..N).rev() {
        let runs = runs.clone();
        let below = top;
        top = rt.create_query_with(
            move |rt| {
                runs[i].fetch_add(1, SeqCst);
                rt.get(below) + 1
            },
            QueryOpts {
                replay: Replay::Allowed,
                ..Default::default()
            },
        );
    }
    assert_eq!(rt.get(top), N as u64);
    let max = runs.iter().map(|r| r.load(SeqCst)).max().unwrap();
    assert!(max <= 2, "a node ran {max} times");
}

fn opts<T>(cycle: CycleStrategy<T>, replay: Replay) -> QueryOpts<T> {
    QueryOpts { cycle, replay }
}

struct IfForm {
    f: Incr<u64>,
    s: Incr<u64>,
    u: Incr<u64>,
}

/// F = IF(A, S, U) with input a = false, and S and U Fallback members
/// reading each other. A's placeholder is true, so at budget 1 F's doomed
/// run reads S on a placeholder, and the speculative chain from S is
/// dropped.
fn if_form(rt: &Runtime<Local>, replay: Replay) -> IfForm {
    let a = rt.create_input(false);
    let cond = rt.create_query_with(move |rt| rt.get(a), opts(CycleStrategy::None, replay));
    let u_cell: Arc<OnceLock<Incr<u64>>> = Arc::new(OnceLock::new());
    let u_for_s = u_cell.clone();
    let s = rt.create_query_with(
        move |rt| rt.get(*u_for_s.get().unwrap()) + 1,
        opts(CycleStrategy::fallback(|_| 100), replay),
    );
    let u = rt.create_query_with(
        move |rt| rt.get(s) + 10,
        opts(CycleStrategy::fallback(|_| 200), replay),
    );
    u_cell.set(u).ok().unwrap();
    let f = rt.create_query_with(
        move |rt| if rt.get(cond) { rt.get(s) } else { rt.get(u) },
        opts(CycleStrategy::None, replay),
    );
    IfForm { f, s, u }
}

/// With every query replayable and every read type holding a placeholder,
/// the speculative chain drops cooperatively: no engine unwind, which
/// `panic = abort` would turn into an abort, and the values are the
/// unbounded run's.
#[wasm_bindgen_test]
fn speculative_chain_drops_without_unwinding() {
    let native: Runtime<Local> = Runtime::new();
    let want = if_form(&native, Replay::Native);
    let want = (native.get(want.f), native.get(want.s), native.get(want.u));

    let rt: Runtime<Local> = Runtime::new();
    rt.set_placeholder(true);
    rt.set_placeholder(0_u64);
    rt.set_max_frames(1);
    let got = if_form(&rt, Replay::Allowed);
    assert_eq!((rt.get(got.f), rt.get(got.s), rt.get(got.u)), want);
}
