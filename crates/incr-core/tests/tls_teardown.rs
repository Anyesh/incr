//! A runtime kept in a `thread_local!` is dropped by that thread's TLS
//! destructors, in no fixed order with the engine's own thread-locals.

use incr_core::{Cells, Local, Runtime, Shared};

thread_local! {
    static LOCAL_RT: Runtime<Local> = Runtime::new();
    static SHARED_RT: Runtime<Shared> = Runtime::new();
    static BUDGETED_RT: Runtime<Local> = {
        let rt = Runtime::new();
        rt.set_max_frames(2);
        rt
    };
}

fn use_rt<C: Cells>(rt: &Runtime<C>) {
    let i = rt.create_input(1_u64);
    let a = rt.create_query(move |rt: &Runtime<C>| rt.get(i) + 1);
    let b = rt.create_query(move |rt: &Runtime<C>| rt.get(a) + 1);
    assert_eq!(rt.get(b), 3);
    rt.set(i, 5);
    assert_eq!(rt.get(b), 7);
}

#[test]
fn local_runtime_in_a_thread_local_drops_at_thread_exit() {
    std::thread::spawn(|| LOCAL_RT.with(use_rt)).join().unwrap();
}

#[test]
fn shared_runtime_in_a_thread_local_drops_at_thread_exit() {
    std::thread::spawn(|| SHARED_RT.with(use_rt))
        .join()
        .unwrap();
}

#[test]
fn budgeted_runtime_in_a_thread_local_drops_at_thread_exit() {
    std::thread::spawn(|| BUDGETED_RT.with(use_rt))
        .join()
        .unwrap();
}
