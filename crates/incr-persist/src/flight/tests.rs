use std::cell::Cell;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

use super::{join_registered, register_in, Control, Flight, Inbox, Verdict, WORKERS};
use crate::{current_value_scope, ValueScope};

const LONG: Option<Duration> = Some(Duration::from_secs(30));

enum Message {
    Add(u64),
    Seal,
}

fn summing(inbox: Inbox<Message>) -> Verdict<u64> {
    let mut sum = 0;
    loop {
        match inbox.recv() {
            Some(Message::Add(n)) => sum += n,
            Some(Message::Seal) => return Verdict::Committed(sum),
            None => return Verdict::Cancelled,
        }
    }
}

#[test]
fn the_worker_gets_every_message_and_posts_its_verdict() {
    let (flight, _token) = Flight::start(None, summing).unwrap();
    for n in 1..=10 {
        flight.send(Message::Add(n));
    }
    flight.send(Message::Seal);
    assert!(flight.wait(LONG));
    assert!(flight.has_verdict());
    assert!(matches!(
        flight.take_verdict(),
        Some(Verdict::Committed(55))
    ));
    assert!(flight.take_verdict().is_none());
}

#[test]
fn the_token_names_the_worker_thread() {
    let (flight, token) =
        Flight::<(), _>::start(None, |_| Verdict::Committed(thread::current().id())).unwrap();
    assert!(flight.wait(LONG));
    match flight.take_verdict() {
        Some(Verdict::Committed(worker)) => assert_eq!(token.worker(), Some(worker)),
        _ => panic!("expected a committed verdict"),
    }
}

#[test]
fn wait_with_a_timeout_returns_false_before_the_verdict() {
    let (flight, _token) = Flight::start(None, summing).unwrap();
    assert!(!flight.wait(Some(Duration::from_millis(20))));
    assert!(!flight.has_verdict());
    flight.send(Message::Seal);
    assert!(flight.wait(LONG));
}

#[test]
fn dropping_the_token_stops_a_waiting_worker() {
    let (flight, token) = Flight::start(None, summing).unwrap();
    flight.send(Message::Add(1));
    drop(token);
    assert!(flight.wait(LONG));
    assert!(flight.control().abandoned());
    assert!(!flight.control().cancelled());
    assert!(matches!(flight.take_verdict(), Some(Verdict::Cancelled)));
}

#[test]
fn dropping_the_token_on_another_thread_stops_the_worker() {
    let (flight, token) = Flight::start(None, summing).unwrap();
    thread::spawn(move || drop(token)).join().unwrap();
    assert!(flight.wait(LONG));
    assert!(matches!(flight.take_verdict(), Some(Verdict::Cancelled)));
}

#[test]
fn cancel_stops_a_waiting_worker() {
    let (flight, _token) = Flight::start(None, summing).unwrap();
    flight.cancel();
    assert!(flight.wait(LONG));
    assert!(flight.control().cancelled());
    assert!(!flight.control().abandoned());
    assert!(matches!(flight.take_verdict(), Some(Verdict::Cancelled)));
}

#[test]
fn dropping_the_flight_abandons_it_without_joining() {
    let (tx, rx) = mpsc::channel();
    let (flight, _token) = Flight::<(), ()>::start(None, move |inbox| {
        let stopped = inbox.recv().is_none() && inbox.control().abandoned();
        tx.send(stopped).unwrap();
        Verdict::Cancelled
    })
    .unwrap();
    drop(flight);
    assert!(rx.recv_timeout(LONG.unwrap()).unwrap());
}

#[test]
fn a_panic_in_the_body_is_posted() {
    let (flight, _token) = Flight::<(), ()>::start(None, |_| panic!("worker panicked")).unwrap();
    assert!(flight.wait(LONG));
    match flight.take_verdict() {
        Some(Verdict::Panicked(payload)) => {
            assert_eq!(payload.downcast_ref::<&str>(), Some(&"worker panicked"))
        }
        _ => panic!("expected the worker's panic"),
    }
}

thread_local! {
    static IN_START: Cell<bool> = const { Cell::new(false) };
}

fn marking_start(f: &mut dyn FnMut()) {
    IN_START.with(|flag| flag.set(true));
    f();
    IN_START.with(|flag| flag.set(false));
}

#[test]
fn the_body_runs_inside_the_worker_start_with_the_scope_current() {
    let scope = ValueScope::new(Arc::new(7u32)).starting_workers(marking_start);
    let (flight, _token) = Flight::<(), _>::start(Some(scope), |_| {
        let state = current_value_scope().and_then(|scope| scope.downcast::<u32>());
        Verdict::Committed((
            IN_START.with(Cell::get),
            state.map(|state| *state),
            thread::current().name().map(str::to_owned),
        ))
    })
    .unwrap();
    assert!(flight.wait(LONG));
    match flight.take_verdict() {
        Some(Verdict::Committed((in_start, state, name))) => {
            assert!(in_start, "the body ran outside the worker start");
            assert_eq!(state, Some(7));
            assert_eq!(name.as_deref(), Some("incr-checkpoint"));
        }
        _ => panic!("expected a committed verdict"),
    }
}

fn panicking_start(_: &mut dyn FnMut()) {
    panic!("start panicked");
}

fn skipping_start(_: &mut dyn FnMut()) {}

fn panicking_after_start(f: &mut dyn FnMut()) {
    f();
    panic!("start panicked after the body");
}

#[test]
fn a_start_that_panics_after_the_body_posts_its_panic() {
    let scope = ValueScope::new(Arc::new(())).starting_workers(panicking_after_start);
    let (flight, _token) =
        Flight::<(), ()>::start(Some(scope), |_| Verdict::Committed(())).unwrap();
    assert!(flight.wait(LONG));
    match flight.take_verdict() {
        Some(Verdict::Panicked(payload)) => assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"start panicked after the body")
        ),
        _ => panic!("expected the start's panic"),
    }
}

#[test]
fn a_start_that_panics_or_skips_the_body_posts_a_panic() {
    for start in [panicking_start as fn(&mut dyn FnMut()), skipping_start] {
        let scope = ValueScope::new(Arc::new(())).starting_workers(start);
        let (flight, _token) =
            Flight::<(), ()>::start(Some(scope), |_| Verdict::Committed(())).unwrap();
        assert!(flight.wait(LONG));
        assert!(matches!(flight.take_verdict(), Some(Verdict::Panicked(_))));
    }
}

#[test]
fn joining_cancels_and_waits_for_running_workers() {
    let (flight, _token) = Flight::start(None, |inbox: Inbox<Message>| {
        let verdict = summing(inbox);
        thread::sleep(Duration::from_millis(50));
        verdict
    })
    .unwrap();
    let mine = {
        let mut workers = WORKERS.lock().unwrap_or_else(PoisonError::into_inner);
        let at = workers
            .iter()
            .position(|(control, _)| Arc::ptr_eq(control, &flight.control))
            .expect("a running worker is registered");
        workers.swap_remove(at)
    };
    join_registered(&Mutex::new(vec![mine]));
    assert!(flight.control().cancelled());
    assert!(
        flight.has_verdict(),
        "join returned before the worker ended"
    );
    assert!(matches!(flight.take_verdict(), Some(Verdict::Cancelled)));
}

#[test]
fn the_registry_is_not_locked_while_joining() {
    let registry = Arc::new(Mutex::new(Vec::new()));
    let (release, released) = mpsc::channel::<()>();
    let worker = thread::spawn(move || released.recv().unwrap());
    register_in(&registry, Arc::new(Control::default()), worker);
    let joining = {
        let registry = Arc::clone(&registry);
        thread::spawn(move || join_registered(&registry))
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    let free = loop {
        let emptied = match registry.try_lock() {
            Ok(workers) => workers.is_empty(),
            Err(TryLockError::WouldBlock) => false,
            Err(TryLockError::Poisoned(_)) => panic!("the registry was poisoned"),
        };
        if emptied || Instant::now() >= deadline {
            break emptied;
        }
        thread::sleep(Duration::from_millis(1));
    };
    release.send(()).unwrap();
    joining.join().unwrap();
    assert!(free, "the registry stayed locked while a worker was joined");
}

/// Recurses until the stack below `top` is `bytes` deep, whatever the
/// frame size of the build.
fn deep(top: usize, bytes: usize) -> usize {
    let frame = std::hint::black_box([0u8; 1024]);
    let here = frame.as_ptr() as usize;
    if top.saturating_sub(here) >= bytes {
        return here;
    }
    std::hint::black_box(deep(top, bytes)) ^ frame[1023] as usize
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_worker_has_the_stack_of_a_main_thread() {
    let (flight, _token) = Flight::<(), usize>::start(None, |_| {
        let top = std::hint::black_box(0u8);
        Verdict::Committed(deep(&top as *const u8 as usize, 6 << 20))
    })
    .unwrap();
    assert!(flight.wait(LONG));
    assert!(matches!(flight.take_verdict(), Some(Verdict::Committed(_))));
}
