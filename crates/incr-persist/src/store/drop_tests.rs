use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::keyed_tests::{Probed, PROBED};
use super::{Input, Options, PendingWrite, PersistedCollection, PersistedKeyed, Store};
use crate::{Backend, Cells, Error, Local, MemoryBackend, Shared};

macro_rules! local_and_shared {
    ($($name:ident),* $(,)?) => {
        $(
            mod $name {
                #[test]
                fn local() {
                    super::$name::<crate::Local>();
                }

                #[test]
                fn shared() {
                    super::$name::<crate::Shared>();
                }
            }
        )*
    };
}

local_and_shared!(
    no_user_drop_runs_under_a_store_lock,
    a_store_call_from_a_subscriber_is_reentrant,
);

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

fn p(text: &str) -> Probed {
    Probed::new(text, false)
}

/// Every public write, apply and fold path of a store and a staged batch,
/// with `Probed` collection elements, keyed rows and keys, and on `Local`
/// input values (a `Shared` input's displaced value is reclaimed by the
/// engine later, on any thread). Undeclare is left out: the engine drops
/// a deleted node's values inside the delete.
fn no_user_drop_runs_under_a_store_lock<C: Cells>() {
    let log = MemoryBackend::new();
    let a = open::<C, _>(log.clone());
    let with_inputs = C::SINGLE_THREADED;
    PROBED.set((0, 0));

    let col: PersistedCollection<Probed, C> = a.collection("c").unwrap();
    let kr: PersistedKeyed<u32, Probed, C> = a.keyed_collection("k").unwrap();
    let kk: PersistedKeyed<Probed, Probed, C> = a.keyed_collection("kk").unwrap();
    let input: Option<Input<Probed>> = with_inputs.then(|| a.input("i", p("d0")).unwrap());
    let _deps = a.open_dep_index().unwrap();

    col.insert(&a, p("c1")).unwrap();
    col.insert(&a, p("c1")).unwrap();
    col.insert(&a, p("c2")).unwrap();
    col.delete(&a, &p("c1")).unwrap();
    col.replace(&a, Some(&p("c2")), p("c3")).unwrap();
    col.replace(&a, None, p("c4")).unwrap();
    a.upsert(&kr, 1, p("r1")).unwrap();
    a.upsert(&kr, 1, p("r2")).unwrap();
    a.upsert(&kr, 2, p("r3")).unwrap();
    a.remove(&kr, &2).unwrap();
    a.reconcile(&kr, [(3, p("r4")), (1, p("r2"))]).unwrap();
    a.upsert(&kk, p("k1"), p("v1")).unwrap();
    a.remove(&kk, &p("k1")).unwrap();
    a.upsert(&kk, p("k2"), p("v2")).unwrap();
    if let Some(input) = input {
        a.set(input, p("i1")).unwrap();
        a.set(input, p("i1")).unwrap();
        a.delete(input).unwrap();
        a.set(input, p("i2")).unwrap();
    }
    a.batch(|batch| {
        if let Some(input) = input {
            batch.set(input, p("i3"));
        }
        batch.insert(&col, p("c5"));
        batch.delete(&col, &p("c5"));
        batch.replace(&col, Some(&p("c4")), p("c6"));
    })
    .unwrap();

    let mut pending = PendingWrite::new();
    if let Some(input) = input {
        pending.set(input, p("i4"));
    }
    pending.insert(&col, p("c7"));
    pending.delete(&col, p("c7"));
    pending.replace(&col, None, p("c8"));
    pending.upsert(&kr, 4, p("r5"));
    pending.remove(&kr, &3);
    pending.reconcile(&kk, [(p("k3"), p("v3"))]);
    a.write_pending(pending).unwrap();

    {
        let mut staged = a.staged_batch().unwrap();
        if let Some(input) = input {
            staged.set(input, p("i5")).unwrap();
        }
        staged.insert(&col, p("c9")).unwrap();
        staged.delete(&col, p("c9")).unwrap();
        staged.replace(&col, None, p("c10")).unwrap();
        let mut pending = PendingWrite::new();
        if let Some(input) = input {
            pending.set(input, p("i6"));
        }
        pending.insert(&col, p("c11"));
        pending.delete(&col, p("c11"));
        pending.replace(&col, Some(p("c10")), p("c12"));
        pending.upsert(&kr, 5, p("r6"));
        pending.remove(&kr, &4);
        pending.reconcile(&kk, [(p("k4"), p("v4"))]);
        staged.write_pending(pending).unwrap();
        staged.commit().unwrap();
    }

    let b = open::<C, _>(log.clone());
    let bcol: PersistedCollection<Probed, C> = b.collection("c").unwrap();
    let bkr: PersistedKeyed<u32, Probed, C> = b.keyed_collection("k").unwrap();
    let bkk: PersistedKeyed<Probed, Probed, C> = b.keyed_collection("kk").unwrap();
    let binput: Option<Input<Probed>> = with_inputs.then(|| b.input("i", p("d1")).unwrap());
    let _bdeps = b.open_dep_index().unwrap();

    a.upsert(&kr, 1, p("r7")).unwrap();
    col.insert(&a, p("c13")).unwrap();
    a.upsert(&kk, p("k4"), p("v5")).unwrap();
    if let Some(input) = input {
        a.set(input, p("i7")).unwrap();
    }
    b.refresh().unwrap();
    a.checkpoint().unwrap();
    a.upsert(&kr, 6, p("r8")).unwrap();
    col.delete(&a, &p("c13")).unwrap();
    a.remove(&kk, &p("k4")).unwrap();
    if let Some(input) = input {
        a.set(input, p("i8")).unwrap();
    }
    b.refresh().unwrap();
    b.upsert(&bkr, 7, p("r9")).unwrap();
    bcol.insert(&b, p("c14")).unwrap();
    b.checkpoint().unwrap();
    a.restored_keys("").unwrap();
    a.refresh().unwrap();

    let mut pending = PendingWrite::new();
    pending.upsert(&kr, 1, p("r10"));
    pending.reconcile(&kk, [(p("k5"), p("v6"))]);
    b.upsert(&bkr, 1, p("r11")).unwrap();
    b.upsert(&bkk, p("k6"), p("v7")).unwrap();
    a.write_pending(pending).unwrap();
    b.refresh().unwrap();

    a.delete(&col).unwrap();
    a.delete(&kr).unwrap();
    if let Some(input) = input {
        a.delete(input).unwrap();
    }
    b.refresh().unwrap();

    {
        let mut scope = a.unlogged_scope().unwrap();
        if let Some(input) = input {
            let handle = scope.input(input).unwrap();
            scope.touched(input).unwrap();
            let held = a.runtime().get(handle);
            a.runtime().set(handle, p("t1"));
            a.runtime().set(handle, held);
        }
        scope.touched(&col).unwrap();
        scope.close().unwrap();
    }
    a.checkpoint().unwrap();

    if let Some(input) = binput {
        assert!(matches!(a.set(input, p("x1")), Err(Error::UnknownHandle)));
    }
    assert!(matches!(
        bcol.insert(&a, p("x2")),
        Err(Error::UnknownHandle)
    ));
    let mut pending = PendingWrite::new();
    pending.insert(&bcol, p("x3"));
    assert!(a.write_pending(pending).is_err());

    let (all, under) = PROBED.get();
    assert!(all > 0, "the writes dropped no Probed value");
    assert_eq!(under, 0, "{under} of {all} drops ran under a lock");
}

type StoreCall = Box<dyn Fn() -> Result<(), Error>>;

thread_local! {
    static REENTER: RefCell<Option<StoreCall>> =
        const { RefCell::new(None) };
    static OUTCOMES: RefCell<Vec<Outcome>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Outcome {
    Ok,
    Reentrant,
    Other(String),
}

/// A value whose `Drop` calls the store armed in `REENTER` and records
/// what the call returned.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Calls(u32);

impl Drop for Calls {
    fn drop(&mut self) {
        if let Some(outcome) = call_armed() {
            OUTCOMES.with(|outcomes| outcomes.borrow_mut().push(outcome));
        }
    }
}

fn arm<C: Cells>(store: &Rc<Store<C, MemoryBackend>>) {
    let store: Weak<Store<C, MemoryBackend>> = Rc::downgrade(store);
    REENTER.with(|reenter| {
        *reenter.borrow_mut() = Some(Box::new(move || {
            store
                .upgrade()
                .map_or(Ok(()), |store| store.stats().map(drop))
        }));
    });
}

fn call_armed() -> Option<Outcome> {
    REENTER.with(|reenter| {
        reenter.borrow().as_ref().map(|call| match call() {
            Ok(()) => Outcome::Ok,
            Err(Error::Reentrant) => Outcome::Reentrant,
            Err(e) => Outcome::Other(e.to_string()),
        })
    })
}

fn outcomes_of(f: impl FnOnce()) -> Vec<Outcome> {
    OUTCOMES.with(|outcomes| outcomes.borrow_mut().clear());
    f();
    OUTCOMES.with(|outcomes| outcomes.take())
}

/// Runs `scenario` on its own thread and fails if it does not finish, so a
/// store call that deadlocks fails the test instead of hanging it.
fn within_watchdog<R: Send + 'static>(scenario: impl FnOnce() -> R + Send + 'static) -> R {
    let (done, finished) = mpsc::channel();
    thread::spawn(move || {
        let _ = done.send(scenario());
    });
    finished
        .recv_timeout(Duration::from_secs(60))
        .expect("the scenario hung or panicked")
}

#[test]
fn a_row_drop_calling_its_store_completes_and_is_refused_only_inside_undeclare() {
    let results = within_watchdog(|| {
        let log = MemoryBackend::new();
        let a = Rc::new(open::<Local, _>(log.clone()));
        let b = open::<Local, _>(log.clone());
        let ka: PersistedKeyed<u32, Calls, Local> = a.keyed_collection("k").unwrap();
        let kb: PersistedKeyed<u32, Calls, Local> = b.keyed_collection("k").unwrap();
        let input: Input<Calls> = a.input("i", Calls(0)).unwrap();
        a.upsert(&ka, 1, Calls(1)).unwrap();
        a.upsert(&ka, 2, Calls(2)).unwrap();
        a.set(input, Calls(3)).unwrap();
        arm(&a);
        let replace = outcomes_of(|| {
            a.upsert(&ka, 1, Calls(4)).unwrap();
        });
        let refresh = outcomes_of(|| {
            b.refresh().unwrap();
            b.upsert(&kb, 2, Calls(5)).unwrap();
            a.refresh().unwrap();
        });
        let reconcile = outcomes_of(|| {
            a.reconcile(&ka, [(1, Calls(4))]).unwrap();
        });
        let checkpoint = outcomes_of(|| {
            a.checkpoint().unwrap();
        });
        let undeclare = outcomes_of(|| {
            a.undeclare(input).unwrap();
        });
        REENTER.with(|reenter| reenter.borrow_mut().take());
        [replace, refresh, reconcile, checkpoint, undeclare]
    });
    let [replace, refresh, reconcile, checkpoint, undeclare] = results;
    for (name, outcomes) in [
        ("replace", replace),
        ("refresh", refresh),
        ("reconcile", reconcile),
        ("checkpoint", checkpoint),
    ] {
        assert!(!outcomes.is_empty(), "{name} dropped no row");
        assert!(
            outcomes.iter().all(|outcome| *outcome == Outcome::Ok),
            "{name}: {outcomes:?}"
        );
    }
    assert!(
        undeclare.contains(&Outcome::Reentrant),
        "undeclare: {undeclare:?}"
    );
    assert!(
        undeclare
            .iter()
            .all(|outcome| matches!(outcome, Outcome::Ok | Outcome::Reentrant)),
        "undeclare: {undeclare:?}"
    );
}

fn a_store_call_from_a_subscriber_is_reentrant<C: Cells>() {
    let store = Rc::new(open::<C, _>(MemoryBackend::new()));
    let input = store.input("n", 0u32).unwrap();
    store
        .subscribe(|_, _, _| {
            if let Some(outcome) = call_armed() {
                OUTCOMES.with(|outcomes| outcomes.borrow_mut().push(outcome));
            }
        })
        .unwrap();
    arm(&store);
    let outcomes = outcomes_of(|| {
        store.set(input, 1).unwrap();
    });
    REENTER.with(|reenter| reenter.borrow_mut().take());
    assert_eq!(outcomes, [Outcome::Reentrant]);
    assert!(store.stats().is_ok(), "the store is usable after");
}

#[test]
fn another_thread_waits_for_a_store_call_and_succeeds() {
    let store = Arc::new(open::<Shared, _>(MemoryBackend::new()));
    let input = store.input("n", 0u32).unwrap();
    let started = Arc::new(AtomicBool::new(false));
    let (spawned, waiter) = mpsc::channel();
    let weak = Arc::downgrade(&store);
    let flag = Arc::clone(&started);
    store
        .subscribe(move |_, _, _| {
            let Some(store) = weak.upgrade() else {
                return;
            };
            let started = Arc::clone(&flag);
            let handle = thread::spawn(move || {
                started.store(true, Ordering::SeqCst);
                store.stats().map(|stats| stats.position)
            });
            while !flag.load(Ordering::SeqCst) {
                thread::yield_now();
            }
            spawned.send(handle).unwrap();
        })
        .unwrap();
    let at = store.set(input, 1).unwrap();
    let handle = waiter.recv().unwrap();
    assert!(started.load(Ordering::SeqCst));
    assert_eq!(handle.join().unwrap().unwrap(), at);
}
