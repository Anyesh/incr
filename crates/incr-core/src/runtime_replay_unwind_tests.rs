//! T8 replay tests of what interrupts a replay: a `set` between its
//! rounds, panics and cycle unwinds out of its entries, the doomed read's
//! node, the entry guard around the driver loop, and pending handles.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::SeqCst};
use std::sync::{Arc, Mutex, OnceLock};

use super::replay_impl::DeliverySeen;
use super::{NodeId, ParkPoint, Runtime};
use crate::cells::{Cells, Local, Shared};
use crate::cycle::{CycleStrategy, Fixpoint, QueryOpts, Replay};
use crate::cycle_state;
use crate::handle::{Incr, RuntimeId};
use crate::runtime::ReadKind;
use crate::state::NodeState;
use crate::table::NodeTable;

macro_rules! local_and_shared {
    ($body:ident, $local:ident, $shared:ident) => {
        #[test]
        fn $local() {
            $body::<Local>();
        }

        #[test]
        fn $shared() {
            $body::<Shared>();
        }
    };
}

fn replayable<T>() -> QueryOpts<T> {
    QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    }
}

fn fallback(value: u64) -> QueryOpts<u64> {
    QueryOpts {
        replay: Replay::Allowed,
        cycle: CycleStrategy::fallback(move |_| value),
    }
}

/// A runtime at frame budget `frames`, unbounded for `None`, with `u64`
/// and `bool` placeholders.
fn runtime<C: Cells>(frames: Option<u32>) -> Runtime<C> {
    let rt = Runtime::<C>::new();
    rt.set_placeholder(0_u64);
    rt.set_placeholder(true);
    rt.set_max_frames(frames.unwrap_or(u32::MAX));
    rt
}

/// A forward reference to a node created later.
type Later = Arc<OnceLock<Incr<u64>>>;

fn later(l: &Later) -> Incr<u64> {
    *l.get().expect("created before the first read")
}

/// A runtime a park hook captures, which its `'static` bound otherwise
/// keeps out of reach.
struct RtRef<C: Cells>(*const Runtime<C>);

// SAFETY: each test installs the hook on the runtime it points to, so the
// runtime outlives every call, and a `Local` runtime only parks on the
// test's thread.
unsafe impl<C: Cells> Send for RtRef<C> {}
// SAFETY: as above.
unsafe impl<C: Cells> Sync for RtRef<C> {}

impl<C: Cells> RtRef<C> {
    fn rt(&self) -> &Runtime<C> {
        // SAFETY: as above.
        unsafe { &*self.0 }
    }
}

fn message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(s) => (*s).to_owned(),
            Err(_) => "<non-string panic payload>".to_owned(),
        },
    }
}

/// What a read returned: a value, a cycle's participants, or a panic's
/// message.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Out {
    Value(u64),
    Cycle(Vec<u32>),
    Panic(String),
}

fn get<C: Cells>(rt: &Runtime<C>, h: Incr<u64>) -> Out {
    match catch_unwind(AssertUnwindSafe(|| rt.get(h))) {
        Ok(v) => Out::Value(v),
        Err(p) => Out::Panic(message(p)),
    }
}

fn try_get<C: Cells>(rt: &Runtime<C>, h: Incr<u64>) -> Out {
    match catch_unwind(AssertUnwindSafe(|| rt.try_get(h))) {
        Ok(Ok(v)) => Out::Value(v),
        Ok(Err(c)) => Out::Cycle(c.participants().iter().map(|p| p.slot()).collect()),
        Err(p) => Out::Panic(message(p)),
    }
}

/// A `set` made by a closure the replay runs between a virtual hit and the
/// pop of the hit's root changes a member's reads, so the episode's
/// validation dissolves it, as in the unbounded run, where the same
/// closure makes the `set` between the cycle's hit and the root's pop.
/// F (Fallback) reads X, X reads Y and then Z, Y reads F while input s is
/// 0, and Z sets s to 1 on its first run. At budget 1 Y's read of F is a
/// virtual hit, and Z, pushed by X's doomed run, makes the `set` before
/// F's pop runs the episode.
fn set_during_replay_dissolves<C: Cells>() {
    let run = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let s = rt.create_input(0_u64);
        let setting = Arc::new(AtomicBool::new(false));
        let set_once = setting.clone();
        let z = rt.create_query_with(
            move |rt| {
                if !set_once.swap(true, SeqCst) {
                    rt.set(s, 1);
                }
                10
            },
            replayable(),
        );
        let f_later = Later::default();
        let fl = f_later.clone();
        let y = rt.create_query_with(
            move |rt| {
                if rt.get(s) == 0 {
                    rt.get(later(&fl)) + 1
                } else {
                    5
                }
            },
            replayable(),
        );
        let x = rt.create_query_with(move |rt| rt.get(y) + rt.get(z), replayable());
        let f = rt.create_query_with(move |rt| rt.get(x), fallback(100));
        f_later.set(f).unwrap();
        let out = get(&rt, f);
        assert_eq!(
            frames.is_some(),
            !rt.dooms.lock().unwrap().is_empty(),
            "a budget replays"
        );
        let dissolved = cycle_state::has_dissolved(rt.runtime_id, f.slot());
        rt.assert_invariant_i();
        assert!(setting.load(SeqCst), "Z made its set");
        // A run that spans the set may land Dirty where a retry made after
        // it lands Clean, so only the settled states are compared.
        let settled: Vec<NodeState> = [f, x, y, z]
            .iter()
            .map(|&h| rt.state_of(h))
            .filter(|s| matches!(s, NodeState::Failed | NodeState::Cyclic))
            .collect();
        let again = get(&rt, f);
        (out, dissolved, settled, again)
    };
    let unbounded = run(None);
    assert_eq!(unbounded.0, Out::Value(15));
    assert!(unbounded.1, "the unbounded episode dissolves");
    for frames in [1, 2] {
        let bounded = run(Some(frames));
        assert_eq!(bounded, unbounded, "at budget {frames}");
    }
}
local_and_shared!(
    set_during_replay_dissolves,
    local_set_during_replay_dissolves,
    shared_set_during_replay_dissolves
);

/// A chain c0 ... c4 whose bottom panics once input p is 1, run at budget
/// 1, where each node is a replay root: the panic, the states it lands and
/// a later read are the unbounded run's, for a first compute and for a
/// recompute.
fn panic_in_a_replay_root<C: Cells>() {
    let run = |frames: Option<u32>, first: bool| {
        let rt = runtime::<C>(frames);
        let p = rt.create_input(u64::from(first));
        let mut below = rt.create_query_with(
            move |rt| {
                if rt.get(p) == 1 {
                    std::panic::panic_any(String::from("root boom"));
                }
                1
            },
            replayable(),
        );
        let mut nodes = vec![below];
        for _ in 0..4 {
            let b = below;
            below = rt.create_query_with(move |rt| rt.get(b) + 1, replayable());
            nodes.push(below);
        }
        if !first {
            assert_eq!(rt.get(below), 5);
            rt.set(p, 1);
        }
        let out = get(&rt, below);
        rt.assert_invariant_i();
        let states: Vec<NodeState> = nodes.iter().map(|&h| rt.state_of(h)).collect();
        (out, states, get(&rt, below))
    };
    for first in [true, false] {
        let unbounded = run(None, first);
        assert_eq!(unbounded.0, Out::Panic("root boom".into()));
        assert_eq!(unbounded.1[0], NodeState::Failed);
        assert_eq!(run(Some(1), first), unbounded, "first compute: {first}");
    }
}
local_and_shared!(
    panic_in_a_replay_root,
    local_panic_in_a_replay_root,
    shared_panic_in_a_replay_root
);

/// P panics on the placeholder its doomed read of N returns: the panic
/// counts as a discard, and P's retry, reading N's real value, lands
/// what the unbounded run does, with nothing Failed.
fn panic_in_a_doomed_frame<C: Cells>() {
    let rt = runtime::<C>(Some(1));
    let n = rt.create_query_with(|_| 5_u64, replayable());
    let runs = Arc::new(AtomicU32::new(0));
    let r = runs.clone();
    let p = rt.create_query_with(
        move |rt| {
            r.fetch_add(1, SeqCst);
            let v = rt.get(n);
            if v == 0 {
                std::panic::panic_any(String::from("doomed boom"));
            }
            v + 1
        },
        replayable(),
    );
    assert_eq!(get(&rt, p), Out::Value(6));
    assert_eq!(runs.load(SeqCst), 2);
    assert_eq!(rt.dooms.lock().unwrap().len(), 1);
    assert_eq!(rt.state_of(p), NodeState::Clean);
    assert_eq!(rt.state_of(n), NodeState::Clean);
    rt.assert_invariant_i();
}
local_and_shared!(
    panic_in_a_doomed_frame,
    local_panic_in_a_doomed_frame,
    shared_panic_in_a_doomed_frame
);

/// F = IF(A, S, R) over a New A whose placeholder is true and whose input
/// is false, so F's doomed run reads S after its placeholder read and S's
/// entry is suspect. `nested` puts the panicking closure in S2, which S
/// reads, so its entry is a non-suspect one a suspect entry's run pushed.
/// With `real_reads` F's real run reads the panicking node too. With
/// `back_edge`, S has a Fallback and S2 reads S before it panics.
struct Speculative {
    out: Out,
    /// F, S and S2.
    states: Vec<NodeState>,
    panicking_runs: u32,
    /// What S2's read of S returned, by the run that went on to panic.
    back_read: Option<u64>,
    /// The held map and S's state at the first entry the driver took
    /// after the panicking run.
    after_panic: Option<(Vec<(RuntimeId, u32)>, NodeState)>,
}

fn speculative<C: Cells>(
    frames: Option<u32>,
    nested: bool,
    real_reads: bool,
    back_edge: bool,
) -> Speculative {
    let rt = runtime::<C>(frames);
    let a_in = rt.create_input(false);
    let a = rt.create_query_with(move |rt| rt.get(a_in), replayable());
    let runs = Arc::new(AtomicU32::new(0));
    let back_read = Arc::new(Mutex::new(None));
    let s_later = Later::default();
    let (r, b, sl) = (runs.clone(), back_read.clone(), s_later.clone());
    let s2 = rt.create_query_with(
        move |rt| -> u64 {
            if back_edge {
                let v = rt.get(later(&sl));
                *b.lock().unwrap() = Some(v);
            }
            r.fetch_add(1, SeqCst);
            std::panic::panic_any(String::from("spec boom"))
        },
        replayable(),
    );
    let s = if nested {
        let opts = if back_edge {
            fallback(50)
        } else {
            replayable()
        };
        rt.create_query_with(move |rt| rt.get(s2) + 1, opts)
    } else {
        s2
    };
    s_later.set(s).unwrap();
    let after_panic = Arc::new(Mutex::new(None));
    let (seen, r, at) = (after_panic.clone(), runs.clone(), RtRef(&rt));
    let installed = rt.park_hook.set(Box::new(move |p| {
        if matches!(p, ParkPoint::BeforeEntry(_)) && r.load(SeqCst) > 0 {
            let mut seen = seen.lock().unwrap();
            if seen.is_none() {
                *seen = Some((cycle_state::held_keys(), at.rt().state_of(s)));
            }
        }
    }));
    assert!(installed.is_ok(), "park hook installed twice");
    let r = rt.create_query_with(|_| 7_u64, replayable());
    let f = rt.create_query_with(
        move |rt| {
            if rt.get(a) {
                rt.get(s)
            } else if real_reads {
                rt.get(s2)
            } else {
                rt.get(r)
            }
        },
        replayable(),
    );
    let out = get(&rt, f);
    rt.assert_invariant_i();
    let back_read = *back_read.lock().unwrap();
    let after_panic = after_panic.lock().unwrap().take();
    Speculative {
        out,
        states: [f, s, s2].iter().map(|&h| rt.state_of(h)).collect(),
        panicking_runs: runs.load(SeqCst),
        back_read,
        after_panic,
    }
}

fn panic_in_a_speculative_entry<C: Cells>() {
    for nested in [false, true] {
        let quiet = speculative::<C>(Some(1), nested, false, false);
        assert_eq!(quiet.out, Out::Value(7), "nested: {nested}");
        assert_eq!(
            quiet.panicking_runs, 1,
            "the speculative entry ran, nested: {nested}"
        );
        for state in &quiet.states[1..] {
            assert!(
                matches!(state, NodeState::New | NodeState::Dirty),
                "a speculative node is {state:?}, nested: {nested}"
            );
        }
        assert_eq!(
            speculative::<C>(None, nested, false, false).out,
            Out::Value(7),
            "nested: {nested}"
        );

        let unbounded = speculative::<C>(None, nested, true, false);
        assert_eq!(unbounded.out, Out::Panic("spec boom".into()));
        let bounded = speculative::<C>(Some(1), nested, true, false);
        assert_eq!(bounded.out, unbounded.out, "nested: {nested}");
        assert_eq!(bounded.states[0], unbounded.states[0], "nested: {nested}");
        assert_eq!(bounded.states[2], unbounded.states[2], "nested: {nested}");
    }
}
local_and_shared!(
    panic_in_a_speculative_entry,
    local_panic_in_a_speculative_entry,
    shared_panic_in_a_speculative_entry
);

/// F5 (a): in the nested form S has a Fallback and S2, the non-suspect
/// entry S's run pushes, reads S, a virtual hit that claims S and returns
/// its seed, then panics. The swallow leaves the held map empty and S
/// unclaimed for the driver's next entry, and F lands the unbounded value
/// with nothing Failed.
fn virtual_hit_then_panic_in_a_speculative_entry<C: Cells>() {
    let unbounded = speculative::<C>(None, true, false, true);
    assert_eq!(unbounded.out, Out::Value(7));
    let bounded = speculative::<C>(Some(1), true, false, true);
    assert_eq!(bounded.out, unbounded.out);
    assert_eq!(bounded.panicking_runs, 1, "S2's run reached its panic");
    assert_eq!(
        bounded.back_read,
        Some(50),
        "S2's read of S was a seeded hit"
    );
    let (held, s_state) = bounded.after_panic.expect("an entry followed the panic");
    assert!(held.is_empty(), "held after the swallow: {held:?}");
    assert!(
        matches!(s_state, NodeState::New | NodeState::Dirty),
        "S is {s_state:?} after the swallow"
    );
    for state in &bounded.states[1..] {
        assert!(
            matches!(state, NodeState::New | NodeState::Dirty),
            "a speculative node is {state:?}"
        );
    }
}
local_and_shared!(
    virtual_hit_then_panic_in_a_speculative_entry,
    local_virtual_hit_then_panic_in_a_speculative_entry,
    shared_virtual_hit_then_panic_in_a_speculative_entry
);

/// A cycle value whose drop panics "drop boom" when `marked`; a clone is
/// unmarked, so only the iterate a step marked panics.
struct Marked {
    v: u64,
    marked: bool,
}

impl Clone for Marked {
    fn clone(&self) -> Self {
        Marked {
            v: self.v,
            marked: false,
        }
    }
}

impl PartialEq for Marked {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Drop for Marked {
    fn drop(&mut self) {
        if self.marked && !std::thread::panicking() {
            std::panic::panic_any(String::from("drop boom"));
        }
    }
}

fn marked(v: u64) -> Marked {
    Marked { v, marked: false }
}

/// The episode callback that panics.
#[derive(Copy, Clone, Debug)]
enum Callback {
    /// A Fallback's `g`, "g boom".
    G,
    /// A Fixpoint's `step`, "step boom".
    Step,
    /// The drop of a Fixpoint's s_1 when s_2 replaces it, "drop boom".
    Drop,
}

/// What the IF form over a panicking episode saw.
#[derive(Debug, PartialEq, Eq)]
struct EpisodePanic {
    out: Out,
    /// S and T after the `get`.
    states: Vec<NodeState>,
}

/// F = IF(A, S, R) as in [`speculative`], where S, suspect, and T read
/// each other, both seeded with `callback`'s strategy. With `real_reads`
/// F's real run reads S. Returns the outcome and, if the callback
/// panicked inside a speculative entry's run, S's and T's states at the
/// first entry the driver took after it.
fn speculative_episode<C: Cells>(
    frames: Option<u32>,
    callback: Callback,
    real_reads: bool,
) -> (EpisodePanic, Option<Vec<NodeState>>) {
    let rt = runtime::<C>(frames);
    rt.set_placeholder(marked(0));
    let panicked = Arc::new(AtomicBool::new(false));
    let strategy = |mark: bool| -> QueryOpts<Marked> {
        let p = panicked.clone();
        let cycle = match callback {
            Callback::G => CycleStrategy::fallback(move |_| -> Marked {
                p.store(p.load(SeqCst) || crate::replay::speculative(), SeqCst);
                std::panic::panic_any(String::from("g boom"))
            }),
            Callback::Step => CycleStrategy::Fixpoint(Fixpoint::new(|| marked(0)).step(
                move |_, _, _| -> Marked {
                    p.store(p.load(SeqCst) || crate::replay::speculative(), SeqCst);
                    std::panic::panic_any(String::from("step boom"))
                },
            )),
            Callback::Drop => CycleStrategy::Fixpoint(Fixpoint::new(|| marked(0)).step(
                move |_, out: Marked, k| {
                    let mark = mark && k == 1;
                    if mark {
                        p.store(p.load(SeqCst) || crate::replay::speculative(), SeqCst);
                    }
                    Marked {
                        v: out.v,
                        marked: mark,
                    }
                },
            )),
        };
        QueryOpts {
            cycle,
            replay: Replay::Allowed,
        }
    };
    let a_in = rt.create_input(false);
    let a = rt.create_query_with(move |rt| rt.get(a_in), replayable());
    let s_later: Arc<OnceLock<Incr<Marked>>> = Arc::default();
    let sl = s_later.clone();
    let t = rt.create_query_with(
        move |rt| marked((rt.get(*sl.get().unwrap()).v + 1).min(3)),
        strategy(false),
    );
    let s = rt.create_query_with(move |rt| marked((rt.get(t).v + 1).min(3)), strategy(true));
    s_later.set(s).unwrap();
    let r = rt.create_query_with(|_| 7_u64, replayable());
    let f = rt.create_query_with(
        move |rt| {
            if rt.get(a) || real_reads {
                rt.get(s).v
            } else {
                rt.get(r)
            }
        },
        replayable(),
    );
    let after_panic = Arc::new(Mutex::new(None));
    let (seen, p, at) = (after_panic.clone(), panicked.clone(), RtRef(&rt));
    let installed = rt.park_hook.set(Box::new(move |point| {
        if matches!(point, ParkPoint::BeforeEntry(_)) && p.load(SeqCst) {
            let mut seen = seen.lock().unwrap();
            if seen.is_none() {
                let rt = at.rt();
                *seen = Some(vec![rt.state_of(s), rt.state_of(t)]);
            }
        }
    }));
    assert!(installed.is_ok(), "park hook installed twice");
    let out = get(&rt, f);
    rt.assert_invariant_i();
    let seen = EpisodePanic {
        out,
        states: vec![rt.state_of(s), rt.state_of(t)],
    };
    let after_panic = after_panic.lock().unwrap().take();
    assert_eq!(after_panic.is_some(), panicked.load(SeqCst));
    (seen, after_panic)
}

/// F5 (b): a speculative entry's run reaches a cycle whose callback
/// panics. Every member is New or Dirty after the speculative run; F
/// lands 7 when the real run never reads S, and panics with the
/// callback's own payload, as the unbounded run does, when it does. Only
/// `g` runs speculatively, as T's virtual hit seeds S: a speculative
/// entry never runs an episode, so `step` and the drop of s_1 are
/// reached by the real run alone.
fn panicking_episode_in_a_speculative_entry<C: Cells>() {
    for (callback, payload, speculative) in [
        (Callback::G, "g boom", true),
        (Callback::Step, "step boom", false),
        (Callback::Drop, "drop boom", false),
    ] {
        let new_or_dirty = |states: &[NodeState]| {
            states
                .iter()
                .all(|s| matches!(s, NodeState::New | NodeState::Dirty))
        };
        let (quiet, after) = speculative_episode::<C>(Some(1), callback, false);
        assert_eq!(quiet.out, Out::Value(7), "{callback:?}");
        assert!(new_or_dirty(&quiet.states), "{callback:?}: {quiet:?}");
        assert_eq!(after.is_some(), speculative, "{callback:?}");
        if let Some(after) = after {
            assert!(new_or_dirty(&after), "{callback:?}: {after:?}");
        }
        assert_eq!(
            speculative_episode::<C>(None, callback, false).0,
            quiet,
            "{callback:?}"
        );

        let (unbounded, _) = speculative_episode::<C>(None, callback, true);
        assert_eq!(unbounded.out, Out::Panic(payload.into()), "{callback:?}");
        let (bounded, after) = speculative_episode::<C>(Some(1), callback, true);
        assert_eq!(bounded, unbounded, "{callback:?}");
        assert_eq!(after.is_some(), speculative, "{callback:?}");
        if let Some(after) = after {
            assert!(new_or_dirty(&after), "{callback:?}: {after:?}");
        }
    }
}
local_and_shared!(
    panicking_episode_in_a_speculative_entry,
    local_panicking_episode_in_a_speculative_entry,
    shared_panicking_episode_in_a_speculative_entry
);

/// How T reads F in the T/F/U shape.
#[derive(Copy, Clone, Debug)]
enum TRead {
    /// `match try_get(F) { Err(_) => 0, Ok(v) => v }`.
    TryGet,
    Get,
    /// `get(F)` under `catch_unwind`, 0 on `Err`.
    Catch,
}

#[derive(Debug, PartialEq, Eq)]
struct Unwound {
    get: Out,
    try_get: Out,
    /// T, F and U after the `get`.
    states: Vec<NodeState>,
    later: Out,
}

/// T reads F, F reads U with `get`, and U, unseeded, reads F with `get`
/// or, with `u_panics`, panics with its own payload.
fn tfu<C: Cells>(
    frames: Option<u32>,
    t_read: TRead,
    u_panics: bool,
) -> (Unwound, Vec<DeliverySeen>, [u32; 3]) {
    let rt = runtime::<C>(frames);
    let f_later = Later::default();
    let fl = f_later.clone();
    let u = rt.create_query_with(
        move |rt| {
            if u_panics {
                std::panic::panic_any(String::from("u boom"));
            }
            rt.get(later(&fl))
        },
        replayable(),
    );
    let f = rt.create_query_with(move |rt| rt.get(u), replayable());
    f_later.set(f).unwrap();
    let t = rt.create_query_with(
        move |rt| match t_read {
            TRead::TryGet => rt.try_get(f).unwrap_or(0),
            TRead::Get => rt.get(f),
            TRead::Catch => catch_unwind(AssertUnwindSafe(|| rt.get(f))).unwrap_or(0),
        },
        replayable(),
    );
    let got = get(&rt, t);
    rt.assert_invariant_i();
    let states = [t, f, u].iter().map(|&h| rt.state_of(h)).collect();
    let deliveries = std::mem::take(&mut *rt.deliveries.lock().unwrap());
    let seen = Unwound {
        get: got,
        try_get: try_get(&rt, t),
        states,
        later: get(&rt, f),
    };
    rt.assert_invariant_i();
    assert!(
        deliveries.iter().all(|d| d.cycle != u_panics),
        "{deliveries:?}"
    );
    (seen, deliveries, [t.slot(), f.slot(), u.slot()])
}

fn unwind_out_of_an_entry<C: Cells>() {
    let cases = [
        (TRead::TryGet, false),
        (TRead::Get, false),
        (TRead::Catch, false),
        (TRead::Catch, true),
        (TRead::Get, true),
    ];
    for (t_read, u_panics) in cases {
        let (unbounded, ..) = tfu::<C>(None, t_read, u_panics);
        for frames in [1, 2] {
            let (bounded, ..) = tfu::<C>(Some(frames), t_read, u_panics);
            assert_eq!(
                bounded, unbounded,
                "{t_read:?}, U panics: {u_panics}, budget {frames}"
            );
        }
    }

    // (i): T returns 0, F and U are New, and at budget 1 the cycle is
    // delivered twice, each consumed once: re-raised at F's retry's read
    // of U, then returned at T's retry's `try_get(F)`.
    let (i, deliveries, [t, f, u]) = tfu::<C>(Some(1), TRead::TryGet, false);
    assert_eq!(i.get, Out::Value(0));
    assert_eq!(i.states[1..], [NodeState::New, NodeState::New]);
    let delivery = |frame, read, kind| DeliverySeen {
        frame,
        read,
        cycle: true,
        kind,
    };
    assert_eq!(
        deliveries,
        [
            delivery(f, u, ReadKind::Get),
            delivery(t, f, ReadKind::TryGet)
        ]
    );

    // (ii): `get(T)` panics with the cycle message and `try_get(T)`
    // returns the cycle, both naming F and U.
    let (ii, ..) = tfu::<C>(Some(1), TRead::Get, false);
    let Out::Panic(text) = &ii.get else {
        panic!("get(T) returned {:?}", ii.get)
    };
    assert!(text.contains("dependency cycle"), "{text}");
    let mut fu = vec![f, u];
    fu.sort_unstable();
    assert_eq!(ii.try_get, Out::Cycle(fu));

    // (iii): a catching T returns 0 through swallow detection.
    assert_eq!(tfu::<C>(Some(1), TRead::Catch, false).0.get, Out::Value(0));

    // (iv): U's own panic Fails F and U, and T too unless T catches it.
    let failed = |s: &[NodeState]| s.iter().filter(|&&s| s == NodeState::Failed).count();
    let (caught, ..) = tfu::<C>(Some(1), TRead::Catch, true);
    assert_eq!(caught.get, Out::Value(0));
    assert_eq!(caught.states[1..], [NodeState::Failed, NodeState::Failed]);
    let (uncaught, ..) = tfu::<C>(Some(1), TRead::Get, true);
    assert_eq!(uncaught.get, Out::Panic("u boom".into()));
    assert_eq!(failed(&uncaught.states), 3);
    for later in [&caught.later, &uncaught.later] {
        let Out::Panic(text) = later else {
            panic!("a later get(F) returned {later:?}")
        };
        assert!(text.contains("last compute panicked"), "{text}");
    }
}
local_and_shared!(
    unwind_out_of_an_entry,
    local_unwind_out_of_an_entry,
    shared_unwind_out_of_an_entry
);

/// P, New, reads N at budget 1, so the read dooms P. Once N is Dirty, its
/// input changed after a compute, so its verify fails, and once it is
/// New. The doom point finds N released, Dirty or New with
/// `verified_at` 0, never Computing; afterwards N is Clean, having run
/// once, in the driver's run of level 1, and P has the unbounded value.
fn doomed_reads_node<C: Cells>() {
    for dirty in [true, false] {
        let rt = runtime::<C>(Some(1));
        let k = rt.create_input(3_u64);
        let runs = Arc::new(Mutex::new(Vec::new()));
        let r = runs.clone();
        let n = rt.create_query_with(
            move |rt| {
                let level = crate::replay::with(|c| c.last().and_then(|x| x.running));
                r.lock().unwrap().push(level);
                rt.get(k) * 2
            },
            replayable(),
        );
        if dirty {
            assert_eq!(rt.get(n), 6);
            rt.set(k, 4);
            runs.lock().unwrap().clear();
        }
        let p = rt.create_query_with(move |rt| rt.get(n) + 1, replayable());
        assert_eq!(rt.get(p), if dirty { 9 } else { 7 });
        let dooms = rt.dooms.lock().unwrap().clone();
        assert_eq!(dooms.len(), 1, "dirty: {dirty}");
        assert_eq!(dooms[0].node, n.slot());
        if dirty {
            assert_eq!(dooms[0].state, NodeState::Dirty);
        } else {
            assert_eq!(dooms[0].state, NodeState::New);
            assert_eq!(dooms[0].verified_at, 0);
        }
        assert_eq!(rt.state_of(n), NodeState::Clean);
        assert_eq!(*runs.lock().unwrap(), [Some(1)], "dirty: {dirty}");
        rt.assert_invariant_i();
    }
}
local_and_shared!(
    doomed_reads_node,
    local_doomed_reads_node,
    shared_doomed_reads_node
);

/// What a doomed read of a group member saw.
#[derive(Debug, PartialEq, Eq)]
struct GroupDoom {
    p: Out,
    /// N and N2 after the `get`.
    states: [NodeState; 2],
    member_runs: u32,
}

/// The doomed read's N as a Dirty `IN_GROUP` member: N (Fallback) and N2
/// form a landed group over M = x % 2, and a cutoff-only `set` of x leaves
/// both Dirty. P, New, reads N at budget 1, so the read dooms P. The doom
/// point finds N and N2 Dirty, neither claimed, and the held map empty;
/// the driver's run of level 1 verifies the group with neither closure
/// running, and P lands the unbounded value.
fn doomed_read_of_a_group_member<C: Cells>() {
    let run = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let x = rt.create_input(1_u64);
        let m = rt.create_query_with(move |rt| rt.get(x) % 2, replayable());
        let runs = Arc::new(AtomicU32::new(0));
        let n_later = Later::default();
        let (r, nl) = (runs.clone(), n_later.clone());
        let n2 = rt.create_query_with(
            move |rt| {
                r.fetch_add(1, SeqCst);
                rt.get(later(&nl)) + rt.get(m)
            },
            replayable(),
        );
        let r = runs.clone();
        let n = rt.create_query_with(
            move |rt| {
                r.fetch_add(1, SeqCst);
                rt.get(n2)
            },
            fallback(100),
        );
        n_later.set(n).unwrap();
        assert_eq!(rt.get(n), 100);
        assert_eq!(rt.get(n2), 101);
        rt.set(x, 3);
        assert_eq!([rt.state_of(n), rt.state_of(n2)], [NodeState::Dirty; 2]);

        let doom_seen = Arc::new(Mutex::new(Vec::new()));
        let entries = Arc::new(Mutex::new(Vec::new()));
        let (d, e, at) = (doom_seen.clone(), entries.clone(), RtRef(&rt));
        let installed = rt.park_hook.set(Box::new(move |p| {
            let rt = at.rt();
            match p {
                ParkPoint::AfterDoom(node) if node == NodeId(n.slot()) => {
                    d.lock().unwrap().push((
                        rt.state_of(n),
                        rt.state_of(n2),
                        cycle_state::held_keys(),
                    ));
                }
                ParkPoint::BeforeEntry(node) => {
                    let verified = rt.group_verifies.lock().unwrap().len();
                    e.lock().unwrap().push((node, verified));
                }
                _ => {}
            }
        }));
        assert!(installed.is_ok(), "park hook installed twice");
        runs.store(0, SeqCst);
        rt.group_verifies.lock().unwrap().clear();
        let p = rt.create_query_with(move |rt| rt.get(n) + 1, replayable());
        let got = get(&rt, p);
        rt.assert_invariant_i();
        let seen = GroupDoom {
            p: got,
            states: [rt.state_of(n), rt.state_of(n2)],
            member_runs: runs.load(SeqCst),
        };
        let doom_seen = std::mem::take(&mut *doom_seen.lock().unwrap());
        let entries = std::mem::take(&mut *entries.lock().unwrap());
        let verifies = std::mem::take(&mut *rt.group_verifies.lock().unwrap());
        (seen, doom_seen, entries, verifies, [n.slot(), n2.slot()])
    };
    let (unbounded, ..) = run(None);
    assert_eq!(
        unbounded,
        GroupDoom {
            p: Out::Value(101),
            states: [NodeState::Clean; 2],
            member_runs: 0,
        }
    );
    let (bounded, doom_seen, entries, verifies, members) = run(Some(1));
    assert_eq!(bounded, unbounded);
    assert_eq!(
        doom_seen,
        [(NodeState::Dirty, NodeState::Dirty, Vec::new())],
        "the doom point found N and N2 Dirty, unclaimed, with nothing held"
    );
    let at = entries
        .iter()
        .position(|&(node, _)| node == NodeId(members[0]))
        .expect("N's entry ran");
    let from = entries[at].1;
    let to = entries.get(at + 1).map_or(verifies.len(), |&(_, v)| v);
    assert!(
        verifies[from..to].iter().any(|v| members.contains(v)),
        "N's entry verified the group: {verifies:?}, entries {entries:?}"
    );
}
local_and_shared!(
    doomed_read_of_a_group_member,
    local_doomed_read_of_a_group_member,
    shared_doomed_read_of_a_group_member
);

/// A `get` and a `try_get` of the head of a 50-deep chain at budget 1 each
/// run their whole driver loop, about 100 entries, inside the one
/// outermost entry the read opened: one 0 to 1 transition of the entry
/// depth, which stores the stack base, per call, and every entry finds
/// the same context with the head as its level-0 target.
fn entry_guard_spans_the_loop<C: Cells>() {
    let rt = runtime::<C>(Some(1));
    for try_ in [false, true] {
        let input = rt.create_input(0_u64);
        let mut head = input;
        for _ in 0..50 {
            let below = head;
            head = rt.create_query_with(move |rt| rt.get(below) + 1, replayable());
        }
        rt.entries.lock().unwrap().clear();
        let before = cycle_state::outermost_entries();
        let value = if try_ {
            rt.try_get(head).expect("no cycle")
        } else {
            rt.get(head)
        };
        assert_eq!(value, 50);
        assert_eq!(cycle_state::outermost_entries() - before, 1, "try: {try_}");
        let entries = std::mem::take(&mut *rt.entries.lock().unwrap());
        assert!(entries.len() >= 98, "{} entries", entries.len());
        let context = entries[0].context;
        for e in &entries {
            assert_eq!(e.context, context, "one context for the whole loop");
            assert_eq!(e.target, Some(head.slot()), "the target stays");
            assert_eq!(e.levels, e.level as usize + 1);
        }
    }
}
local_and_shared!(
    entry_guard_spans_the_loop,
    local_entry_guard_spans_the_loop,
    shared_entry_guard_spans_the_loop
);

/// P's run dooms on its read of a New N, then misses a table and its own
/// children: each miss returns the pending handle, which reads the
/// placeholder there and creates nothing. P's retry creates the real
/// nodes and lands the unbounded value, and an escaped pending handle
/// panics outside its frame. P reads its input child but not its query
/// child, whose read is `doomed_child_query_read_terminates`'s.
fn doomed_miss_returns_a_pending_handle<C: Cells>()
where
    NodeTable<u64, u64, C>: Send + Sync,
{
    let run = |frames: Option<u32>| {
        let rt = runtime::<C>(frames);
        let table = Arc::new(rt.create_table::<u64, u64>());
        let n = rt.create_query_with(|_| 100_u64, replayable());
        let minted = Arc::new(Mutex::new(Vec::new()));
        let (t, m) = (table.clone(), minted.clone());
        let p = rt.create_query_with(
            move |rt| {
                let base = rt.get(n);
                let row = t.get_or_insert_query(rt, 1, |_| 40);
                let child = rt.child_input("one", || 2_u64);
                let query = rt.child_query("two", |_| 3_u64);
                let got = rt.get(row) + rt.get(child);
                m.lock().unwrap().push((base, [row, child, query], got));
                base + got
            },
            replayable(),
        );
        let nodes = rt.node_count();
        let value = rt.get(p);
        let created = rt.node_count() - nodes;
        let minted = std::mem::take(&mut *minted.lock().unwrap());
        (rt, value, created, minted)
    };
    let (_, unbounded, created, _) = run(None);
    assert_eq!(unbounded, 142);
    let (rt, value, bounded_created, minted) = run(Some(1));
    assert_eq!(value, unbounded);
    assert_eq!(bounded_created, created, "the doomed run creates nothing");
    let (base, doomed, got) = minted[0];
    assert_eq!((base, got), (0, 0), "pending handles read the placeholder");
    assert!(doomed.iter().all(|&h| h == doomed[0] && !rt.is_alive(h)));
    let (base, real, _) = minted[minted.len() - 1];
    assert_eq!(base, 100);
    assert!(real.iter().all(|&h| rt.is_alive(h)), "{minted:?}");
    let text = match catch_unwind(AssertUnwindSafe(|| rt.get(doomed[0]))) {
        Ok(v) => panic!("an escaped pending handle read {v}"),
        Err(p) => message(p),
    };
    assert!(
        text.contains("pending handle used outside the replay attempt"),
        "{text}"
    );
}
local_and_shared!(
    doomed_miss_returns_a_pending_handle,
    local_doomed_miss_returns_a_pending_handle,
    shared_doomed_miss_returns_a_pending_handle
);

/// P creates a query child and reads it, at budget 1: the read of the New
/// child dooms P, and P's retry must find the child and land, as the
/// unbounded run does after one run. The runs are capped so that a retry
/// that never lands fails instead of hanging.
fn doomed_child_query_read_terminates<C: Cells>() {
    let rt = runtime::<C>(Some(1));
    let runs = Arc::new(AtomicU32::new(0));
    let r = runs.clone();
    let p = rt.create_query_with(
        move |rt| {
            let run = r.fetch_add(1, SeqCst) + 1;
            assert!(run <= 8, "P ran {run} times");
            let query = rt.child_query("two", |_| 3_u64);
            rt.get(query) + 1
        },
        replayable(),
    );
    assert_eq!(get(&rt, p), Out::Value(4));
    assert!(runs.load(SeqCst) <= 2, "P ran {} times", runs.load(SeqCst));
}
local_and_shared!(
    doomed_child_query_read_terminates,
    local_doomed_child_query_read_terminates,
    shared_doomed_child_query_read_terminates
);
