use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use super::NodeTable;
use crate::cells::{Cells, Local, Shared};
use crate::child_test::run_in_child;
use crate::cycle::{QueryOpts, Replay};
use crate::handle::Incr;
use crate::node::NodeId;
use crate::reclaim_probe::{self, RetireSite};
use crate::runtime::{DeleteError, Runtime};
use crate::value_code::CONTRACT_MESSAGE;

fn replayable<T>() -> QueryOpts<T> {
    QueryOpts {
        replay: Replay::Allowed,
        ..Default::default()
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(s) => (*s).to_string(),
            Err(_) => "<non-string panic payload>".to_string(),
        },
    }
}

fn panics_with(message: &str, f: impl FnOnce()) {
    let got = panic_message(catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic"));
    assert!(got.contains(message), "expected {message:?}, got {got:?}");
}

macro_rules! both {
    ($($name:ident),* $(,)?) => {$(
        mod $name {
            #[test]
            fn local() {
                super::$name::<crate::cells::Local>(&crate::runtime::Runtime::new());
            }

            #[test]
            fn shared() {
                super::$name::<crate::cells::Shared>(&crate::runtime::Runtime::new());
            }
        }
    )*};
}

both!(
    a_hit_returns_the_first_node,
    the_table_grows_and_keeps_every_entry,
    remove_deletes_the_node_and_its_entry,
    a_deleted_node_is_a_miss_and_is_replaced,
    computes_create_their_nodes_through_tables,
    lookups_refuse_a_compute_of_their_runtime,
    another_runtime_is_refused,
    for_each_visits_each_entry_once_while_its_callback_inserts,
    a_clone_panic_in_rehash_creates_nothing,
    a_key_that_misses_from_eq_hits_the_contract,
    a_key_on_a_lock_free_hit_may_read_a_clean_node,
    a_key_that_misses_under_the_miss_mutex_hits_the_contract,
    a_clone_that_misses_hits_the_contract,
    pending_handles_in_their_doomed_frame,
    pending_handles_out_of_their_frame,
    a_doomed_miss_needs_a_placeholder,
);

fn a_hit_returns_the_first_node<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_table::<String, i64>();
    let a = t.get_or_insert_input(rt, "a".into(), || 1);
    let b = t.get_or_insert_input(rt, "a".into(), || panic!("a hit runs no init"));
    assert_eq!(a, b);
    assert_eq!(rt.get(a), 1);
    let q = t.get_or_insert_query(rt, "a".into(), |_| panic!("a hit creates no query"));
    assert_eq!(q, a, "a hit returns the entry whichever call created it");
    assert_eq!(t.get(rt, &"a".to_string()), Some(a));
    assert_eq!(t.get(rt, &"b".to_string()), None);

    let q = t.get_or_insert_query(rt, "q".into(), |_| 7);
    assert_eq!(rt.get(q), 7);
    assert_eq!(t.get(rt, &"q".to_string()), Some(q));
}

fn the_table_grows_and_keeps_every_entry<C: Cells>(rt: &Runtime<C>) {
    let n = if cfg!(miri) { 100 } else { 1000 };
    let t = rt.create_table::<u64, i64>();
    let handles: Vec<_> = (0..n)
        .map(|i| t.get_or_insert_input(rt, i, || i as i64))
        .collect();
    for (i, h) in handles.iter().enumerate() {
        assert_eq!(t.get(rt, &(i as u64)), Some(*h));
        assert_eq!(rt.get(*h), i as i64);
    }
    let (capacity, live, tombstones) = t.shape();
    assert!(capacity.is_power_of_two());
    assert_eq!((live, tombstones), (n as usize, 0));
    assert!(8 * live <= 7 * capacity, "{live} live in {capacity}");
}

fn remove_deletes_the_node_and_its_entry<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_table::<u64, i64>();
    let h = t.get_or_insert_input(rt, 1, || 10);
    assert_eq!(t.remove(rt, &1), Ok(()));
    assert!(!rt.is_alive(h));
    assert_eq!(t.get(rt, &1), None);
    assert_eq!(t.remove(rt, &1), Err(DeleteError::NotAlive));
    assert_eq!(t.remove(rt, &2), Err(DeleteError::NotAlive));
    assert_eq!((t.shape().1, t.shape().2), (0, 1));

    let h2 = t.get_or_insert_input(rt, 1, || 20);
    assert_ne!(h, h2);
    let q = rt.create_query(move |rt| rt.get(h2) + 1);
    assert_eq!(rt.get(q), 21);
    assert_eq!(t.remove(rt, &1), Err(DeleteError::HasDependents));
    assert_eq!(t.get(rt, &1), Some(h2));
}

fn a_deleted_node_is_a_miss_and_is_replaced<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_table::<u64, i64>();
    let h = t.get_or_insert_input(rt, 1, || 10);
    rt.delete_node(h);
    assert_eq!(t.get(rt, &1), None);
    let h2 = t.get_or_insert_input(rt, 1, || 20);
    assert_ne!(h, h2);
    assert_eq!(rt.get(h2), 20);
    assert_eq!(t.shape().1, 1);
    assert_eq!(t.shape().2, 1, "the miss tombstoned the dead bucket");
}

fn computes_create_their_nodes_through_tables<C: Cells>(rt: &Runtime<C>)
where
    NodeTable<u64, i64, C>: Send + Sync,
{
    let inputs = Arc::new(rt.create_table::<u64, i64>());
    let queries = Arc::new(rt.create_table::<u64, i64>());
    let n = rt.create_input(3_i64);
    let (i2, q2) = (inputs.clone(), queries.clone());
    let sum = rt.create_query(move |rt| {
        (0..rt.get(n) as u64)
            .map(|i| {
                let x = i2.get_or_insert_input(rt, i, || 10 * i as i64);
                rt.get(q2.get_or_insert_query(rt, i, move |rt| rt.get(x) + 1))
            })
            .sum::<i64>()
    });
    assert_eq!(rt.get(sum), 1 + 11 + 21);
    rt.set(n, 4);
    assert_eq!(rt.get(sum), 1 + 11 + 21 + 31);
    let x0 = inputs.get(rt, &0).expect("created by the compute");
    rt.set(x0, 100);
    assert_eq!(rt.get(sum), 101 + 11 + 21 + 31);
}

fn lookups_refuse_a_compute_of_their_runtime<C: Cells>(rt: &Runtime<C>)
where
    NodeTable<u64, i64, C>: Send + Sync,
{
    let t = Arc::new(rt.create_table::<u64, i64>());
    t.get_or_insert_input(rt, 1, || 1);
    let t2 = t.clone();
    let q = rt.create_query(move |rt| {
        let _ = t2.get(rt, &1);
        0_i64
    });
    panics_with("NodeTable::get called during compute", || {
        rt.get(q);
    });
    let t3 = t.clone();
    let q = rt.create_query(move |rt| {
        t3.for_each(rt, |_, _| {});
        0_i64
    });
    panics_with("NodeTable::for_each called during compute", || {
        rt.get(q);
    });
}

fn another_runtime_is_refused<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_table::<u64, i64>();
    let other = Runtime::<C>::new();
    panics_with("used with runtime", || {
        t.get_or_insert_input(&other, 1, || 1);
    });
    panics_with("used with runtime", || {
        let _ = t.remove(&other, &1);
    });
}

fn for_each_visits_each_entry_once_while_its_callback_inserts<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_table::<u64, i64>();
    for i in 0..20 {
        t.get_or_insert_input(rt, i, || i as i64);
    }
    let capacity = t.shape().0;
    let mut seen = Vec::new();
    t.for_each(rt, |k, h| {
        seen.push(*k);
        if *k < 20 {
            assert_eq!(rt.get(h), *k as i64);
            t.get_or_insert_input(rt, 100 + k, || 0);
        }
    });
    assert!(t.shape().0 > capacity, "the callback's inserts rehashed");
    seen.sort_unstable();
    let original: Vec<u64> = seen.iter().copied().filter(|k| *k < 20).collect();
    assert_eq!(original, (0..20).collect::<Vec<_>>());
    let mut dedup = seen.clone();
    dedup.dedup();
    assert_eq!(dedup, seen, "each pair is visited at most once");
    for i in 0..20 {
        assert!(t.get(rt, &(100 + i)).is_some());
    }
}

/// A hook and the number of key calls left before it fires.
type KeyHook = (Cell<u32>, Box<dyn Fn()>);

thread_local! {
    /// A test hook that key methods run, and when: `Some(n)` fires on the
    /// `n`th call from now.
    static KEY_HOOK: RefCell<Option<KeyHook>> = const { RefCell::new(None) };
    static CLONE_PANICS: Cell<bool> = const { Cell::new(false) };
}

fn arm_key_hook(after: u32, hook: impl Fn() + 'static) {
    KEY_HOOK.with(|h| *h.borrow_mut() = Some((Cell::new(after), Box::new(hook))));
}

fn disarm_key_hook() {
    KEY_HOOK.with(|h| *h.borrow_mut() = None);
}

fn run_key_hook() {
    let fire = KEY_HOOK.with(|h| {
        h.borrow().as_ref().is_some_and(|(n, _)| {
            let left = n.get();
            n.set(left.saturating_sub(1));
            left == 1
        })
    });
    if fire {
        let hook = KEY_HOOK.with(|h| h.borrow_mut().take());
        let (_, f) = hook.expect("checked above");
        f();
    }
}

/// A key whose `Eq` and `Clone` run the key hook, and whose `Clone`
/// panics while `CLONE_PANICS` is set.
#[derive(Debug)]
struct HookKey(u64);

impl Hash for HookKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for HookKey {
    fn eq(&self, other: &Self) -> bool {
        run_key_hook();
        self.0 == other.0
    }
}

impl Eq for HookKey {}

impl Clone for HookKey {
    fn clone(&self) -> Self {
        run_key_hook();
        if CLONE_PANICS.with(Cell::get) {
            panic!("clone of HookKey({}) panicked on purpose", self.0);
        }
        HookKey(self.0)
    }
}

/// Fills a fresh table to one entry short of its first rehash.
fn table_at_the_brink<C: Cells>(rt: &Runtime<C>) -> NodeTable<HookKey, i64, C> {
    let t = rt.create_table::<HookKey, i64>();
    for i in 0..14 {
        t.get_or_insert_input(rt, HookKey(i), || i as i64);
    }
    assert_eq!(t.shape(), (16, 14, 0));
    t
}

fn a_clone_panic_in_rehash_creates_nothing<C: Cells>(rt: &Runtime<C>) {
    let t = table_at_the_brink(rt);
    let nodes = rt.node_count();
    CLONE_PANICS.with(|c| c.set(true));
    for _ in 0..if cfg!(miri) { 3 } else { 100 } {
        panics_with("panicked on purpose", || {
            t.get_or_insert_input(rt, HookKey(99), || 99);
        });
        assert_eq!(rt.node_count(), nodes);
        assert_eq!(t.shape(), (16, 14, 0), "the array is untouched");
    }
    CLONE_PANICS.with(|c| c.set(false));
    assert_eq!(t.get(rt, &HookKey(99)), None);
    let h = t.get_or_insert_input(rt, HookKey(99), || 99);
    assert_eq!(rt.get(h), 99);
    assert_eq!(t.shape(), (32, 15, 0));
}

fn a_key_that_misses_from_eq_hits_the_contract<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_table::<HookKey, i64>();
    t.get_or_insert_input(rt, HookKey(1), || 1);
    let (rt_ptr, t_ptr) = (
        rt as *const Runtime<C>,
        &t as *const NodeTable<HookKey, i64, C>,
    );
    arm_key_hook(1, move || {
        // SAFETY: both outlive the hook, which is disarmed below.
        let (rt, t) = unsafe { (&*rt_ptr, &*t_ptr) };
        t.get_or_insert_input(rt, HookKey(2), || 2);
    });
    panics_with(CONTRACT_MESSAGE, || {
        t.get_or_insert_input(rt, HookKey(1), || 1);
    });
    disarm_key_hook();
    let x = rt.create_input(0_i64);
    arm_key_hook(1, move || {
        // SAFETY: as above.
        unsafe { &*rt_ptr }.set(x, 5);
    });
    panics_with(CONTRACT_MESSAGE, || {
        t.get(rt, &HookKey(1));
    });
    disarm_key_hook();
    assert_eq!(rt.get(x), 0);
    let h = t.get_or_insert_input(rt, HookKey(2), || 2);
    assert_eq!(rt.get(h), 2, "the table stays usable");
}

fn a_key_on_a_lock_free_hit_may_read_a_clean_node<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_table::<HookKey, i64>();
    let h = t.get_or_insert_input(rt, HookKey(1), || 1);
    let x = rt.create_input(41_i64);
    let read = Arc::new(Mutex::new(None));
    let (rt_ptr, out) = (rt as *const Runtime<C>, read.clone());
    arm_key_hook(1, move || {
        // SAFETY: the runtime outlives the hook, which fires inside the
        // lookup below.
        *out.lock().unwrap() = Some(unsafe { &*rt_ptr }.get(x));
    });
    assert_eq!(t.get(rt, &HookKey(1)), Some(h));
    assert_eq!(*read.lock().unwrap(), Some(41));
    disarm_key_hook();
}

fn a_key_that_misses_under_the_miss_mutex_hits_the_contract<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_table::<HookKey, i64>();
    let dead = t.get_or_insert_input(rt, HookKey(1), || 1);
    rt.delete_node(dead);
    let (rt_ptr, t_ptr) = (
        rt as *const Runtime<C>,
        &t as *const NodeTable<HookKey, i64, C>,
    );
    // The lock-free probe compares against the dead bucket first; the
    // second compare is the probe under the miss mutex.
    arm_key_hook(2, move || {
        // SAFETY: both outlive the hook, which is disarmed below.
        let (rt, t) = unsafe { (&*rt_ptr, &*t_ptr) };
        t.get_or_insert_input(rt, HookKey(2), || 2);
    });
    panics_with(CONTRACT_MESSAGE, || {
        t.get_or_insert_input(rt, HookKey(1), || 1);
    });
    disarm_key_hook();
    let h = t.get_or_insert_input(rt, HookKey(1), || 3);
    assert_eq!(rt.get(h), 3, "the miss mutex was released");
}

fn a_clone_that_misses_hits_the_contract<C: Cells>(rt: &Runtime<C>) {
    let t = table_at_the_brink(rt);
    let x = rt.create_input(0_i64);
    let rt_ptr = rt as *const Runtime<C>;
    arm_key_hook(1, move || {
        // SAFETY: the runtime outlives the hook, which fires in the
        // rehash below.
        unsafe { &*rt_ptr }.set(x, 1);
    });
    let nodes = rt.node_count();
    panics_with(CONTRACT_MESSAGE, || {
        t.get_or_insert_input(rt, HookKey(99), || 99);
    });
    disarm_key_hook();
    assert_eq!(rt.node_count(), nodes);
    assert_eq!(t.shape(), (16, 14, 0));
    assert_eq!(rt.get(t.get_or_insert_input(rt, HookKey(99), || 99)), 99);
}

/// What a doomed frame saw, recorded for the test to check after it.
#[derive(Default)]
struct Doomed {
    pending: Option<Incr<i64>>,
    second: Option<Incr<i64>>,
    hit: Option<Incr<i64>>,
    log: Vec<String>,
}

fn pending_handles_in_their_doomed_frame<C: Cells>(rt: &Runtime<C>)
where
    NodeTable<u64, i64, C>: Send + Sync,
{
    rt.set_placeholder::<i64>(-1);
    let gate = rt.doom_gate();
    let t = Arc::new(rt.create_table::<u64, i64>());
    let present = t.get_or_insert_input(rt, 1, || 10);
    let seen = Arc::new(Mutex::new(Doomed::default()));
    let (t2, out) = (t.clone(), seen.clone());
    let q = rt.create_query_with(
        move |rt| {
            if rt.get(gate).0 {
                return 0;
            }
            let mut out = out.lock().unwrap();
            out.hit = Some(t2.get_or_insert_input(rt, 1, || 0));
            let p = t2.get_or_insert_input(rt, 2, || 0);
            out.second = Some(t2.get_or_insert_query(rt, 3, |_| 0));
            out.pending = Some(p);
            let mut log = |what: &str, r: std::thread::Result<()>| {
                out.log.push(match r {
                    Ok(()) => format!("{what}: ok"),
                    Err(e) => format!("{what}: {}", panic_message(e)),
                })
            };
            let quiet = |f: &dyn Fn()| catch_unwind(AssertUnwindSafe(f));
            assert_eq!(rt.get(p), -1);
            assert_eq!(rt.try_get(p), Ok(-1));
            assert_eq!(rt.try_delete_node(p), Err(DeleteError::NotAlive));
            assert!(!rt.is_alive(p));
            assert_eq!(rt.cycle_of(p), None);
            log("set", quiet(&|| rt.set(p, 1)));
            log("delete_node", quiet(&|| rt.delete_node(p)));
            log(
                "observe",
                quiet(&|| {
                    rt.observe(p, |_| {});
                }),
            );
            log(
                "delete_with_dependents",
                quiet(&|| {
                    rt.delete_with_dependents(p);
                }),
            );
            0_i64
        },
        replayable(),
    );
    let nodes = rt.node_count();
    assert_eq!(rt.get(q), 0);
    let seen = seen.lock().unwrap();
    assert_eq!(seen.hit, Some(present), "a hit returns the real handle");
    assert_eq!(
        seen.second, seen.pending,
        "pending handles of one runtime are equal"
    );
    assert_eq!(rt.node_count(), nodes, "a doomed miss creates nothing");
    assert_eq!(t.get(rt, &2), None, "and inserts nothing");
    for (what, line) in ["set", "delete_node", "observe"].iter().zip(&seen.log) {
        assert!(
            line.starts_with(what) && line.contains("can only be read"),
            "{line}"
        );
    }
    assert!(seen.log[3].contains("pending handle"), "{}", seen.log[3]);
    let snapshot = rt.graph_snapshot();
    let info = snapshot
        .iter()
        .find(|n| n.id.0 == q.slot())
        .expect("q is in the graph");
    assert_eq!(
        info.dependencies,
        vec![NodeId(gate.slot())],
        "a placeholder read records no dependency"
    );
}

fn pending_handles_out_of_their_frame<C: Cells>(rt: &Runtime<C>)
where
    NodeTable<u64, i64, C>: Send + Sync,
{
    rt.set_placeholder::<i64>(-1);
    let gate = rt.doom_gate();
    let t = Arc::new(rt.create_table::<u64, i64>());
    let escaped = Arc::new(Mutex::new(None));
    let (t2, out) = (t.clone(), escaped.clone());
    let q = rt.create_query_with(
        move |rt| {
            if !rt.get(gate).0 {
                *out.lock().unwrap() = Some(t2.get_or_insert_input(rt, 2, || 0));
            }
            0_i64
        },
        replayable(),
    );
    rt.get(q);
    let p = escaped.lock().unwrap().expect("minted in the frame");
    const OUTSIDE: &str = "pending handle used outside the replay attempt";
    panics_with(OUTSIDE, || {
        rt.get(p);
    });
    panics_with(OUTSIDE, || {
        let _ = rt.try_get(p);
    });
    panics_with(OUTSIDE, || rt.set(p, 1));
    panics_with(OUTSIDE, || rt.delete_node(p));
    panics_with(OUTSIDE, || {
        rt.observe(p, |_| {});
    });
    panics_with(OUTSIDE, || {
        rt.cycle_of(p);
    });
    panics_with("pending handle", || {
        rt.delete_with_dependents(p);
    });
    assert_eq!(rt.try_delete_node(p), Err(DeleteError::NotAlive));
    assert!(!rt.is_alive(p));

    // Inside another frame of this runtime that is not doomed, and inside
    // a doomed frame of another runtime.
    let q = rt.create_query(move |rt| rt.get(p));
    panics_with(OUTSIDE, || {
        rt.get(q);
    });
    let other = Runtime::<C>::new();
    other.set_placeholder::<i64>(-2);
    let other_gate = other.doom_gate();
    let in_doomed = Arc::new(Mutex::new(None));
    let seen = in_doomed.clone();
    let q = other.create_query_with(
        move |other| {
            let doomed = !other.get(other_gate).0;
            let read = catch_unwind(AssertUnwindSafe(|| other.get(p)));
            if doomed {
                *seen.lock().unwrap() = Some(read.err().map(panic_message));
                return 0;
            }
            read.unwrap_or_else(|e| std::panic::resume_unwind(e))
        },
        replayable(),
    );
    panics_with(OUTSIDE, || {
        other.get(q);
    });
    let doomed = in_doomed
        .lock()
        .unwrap()
        .take()
        .expect("the doomed run read p");
    assert!(doomed.is_some_and(|m| m.contains(OUTSIDE)));
}

fn a_doomed_miss_needs_a_placeholder<C: Cells>(rt: &Runtime<C>)
where
    NodeTable<u64, String, C>: Send + Sync,
{
    let gate = rt.doom_gate();
    let t = Arc::new(rt.create_table::<u64, String>());
    let (t2, seen) = (t.clone(), Arc::new(Mutex::new(None)));
    let out = seen.clone();
    let q = rt.create_query_with(
        move |rt| {
            if !rt.get(gate).0 {
                let miss = catch_unwind(AssertUnwindSafe(|| {
                    t2.get_or_insert_input(rt, 1, String::new);
                }));
                *out.lock().unwrap() = miss.err().map(panic_message);
            }
            0_i64
        },
        replayable(),
    );
    assert_eq!(rt.get(q), 0);
    let message = seen
        .lock()
        .unwrap()
        .take()
        .expect("the doomed miss panicked");
    assert!(
        message.contains("needs a placeholder for alloc::string::String"),
        "{message}"
    );
    panics_with("placeholder already set", || {
        rt.set_placeholder::<u8>(1);
        rt.set_placeholder::<u8>(2);
    });
}

/// A key whose `Drop` panics while armed.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DropKey(u64);

static DROP_ARMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

impl Drop for DropKey {
    fn drop(&mut self) {
        if DROP_ARMED
            .compare_exchange(
                self.0,
                u64::MAX,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .is_ok()
        {
            panic!("drop of DropKey({}) panicked on purpose", self.0);
        }
    }
}

/// The old array's keys drop after the miss mutex is released, and a
/// panicking one surfaces from the miss that rehashed; the entry it
/// inserted is in the table.
fn key_drop_panic_at_rehash<C: Cells>(rt: &Runtime<C>) {
    let t = rt.create_table::<DropKey, i64>();
    for i in 0..14 {
        t.get_or_insert_input(rt, DropKey(i), || i as i64);
    }
    DROP_ARMED.store(3, std::sync::atomic::Ordering::SeqCst);
    let got = catch_unwind(AssertUnwindSafe(|| {
        t.get_or_insert_input(rt, DropKey(99), || 99);
    }));
    let message = panic_message(got.expect_err("the key's drop panic surfaces"));
    assert!(message.contains("DropKey(3)"), "{message}");
    assert_eq!(t.shape().0, 32);
    let h = t.get(rt, &DropKey(99)).expect("inserted before the drop");
    assert_eq!(rt.get(h), 99);
    for i in 0..14 {
        assert!(t.get(rt, &DropKey(i)).is_some());
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn local_key_drop_panic_at_rehash_surfaces() {
    run_in_child(
        "table::tests::local_key_drop_panic_at_rehash_surfaces",
        || {
            key_drop_panic_at_rehash(&Runtime::<Local>::new());
        },
    )
    .assert_clean();
}

#[test]
#[cfg_attr(miri, ignore)]
fn shared_key_drop_panic_at_rehash_surfaces() {
    run_in_child(
        "table::tests::shared_key_drop_panic_at_rehash_surfaces",
        || {
            let rt = Runtime::<Shared>::new();
            reclaim_probe::force(RetireSite::TableRetire);
            key_drop_panic_at_rehash(&rt);
        },
    )
    .assert_clean();
}
