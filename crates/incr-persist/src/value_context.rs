use std::any::Any;
use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

/// Which slot a user value is being encoded or decoded for, visible to its
/// `Serialize` and `Deserialize` impls through [`current_value_context`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueContext {
    /// An input or collection of a [`Store`](crate::Store), by its key.
    Key(String),
    /// The key or the value of a memo table's entry.
    MemoTable { table: String, part: MemoPart },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoPart {
    Key,
    Value,
}

/// A memo entry the last checkpoints left out because its key or value did
/// not encode, or did not decode back equal to itself. The entry is
/// recomputed after the next restart; nothing wrong is stored. At most one
/// per table and part is kept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodeMismatch {
    pub table: String,
    pub part: MemoPart,
}

thread_local! {
    static CURRENT: RefCell<Option<ValueContext>> = const { RefCell::new(None) };
}

/// The context of the encode or decode running on this thread, if any.
pub fn current_value_context() -> Option<ValueContext> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Makes `context` current on this thread until dropped, then restores the
/// previous one, also when unwinding.
pub fn enter_value_context(context: ValueContext) -> ValueContextGuard {
    let previous = CURRENT.with(|current| current.replace(Some(context)));
    ValueContextGuard { previous }
}

#[must_use = "the context ends when the guard is dropped"]
pub struct ValueContextGuard {
    previous: Option<ValueContext>,
}

impl Drop for ValueContextGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        let _ = CURRENT.try_with(|current| current.replace(previous));
    }
}

/// A binding's own state for its value code, such as a table of codecs,
/// that a [`Store`](crate::Store) makes current through
/// [`current_value_scope`] while it holds its lock, on whichever thread
/// takes it. Set it in [`Options::value_scope`](crate::Options::value_scope).
#[derive(Clone)]
pub struct ValueScope {
    state: Arc<dyn Any + Send + Sync>,
    hold: Option<fn(&mut dyn FnMut())>,
    idle: Option<fn(&mut dyn FnMut())>,
    wait: Option<fn(&mut dyn FnMut())>,
    start: Option<fn(&mut dyn FnMut())>,
}

impl ValueScope {
    pub fn new<T: Any + Send + Sync>(state: Arc<T>) -> Self {
        ValueScope {
            state,
            hold: None,
            idle: None,
            wait: None,
            start: None,
        }
    }

    /// This scope with `hold`, which runs a batch of value calls (clones,
    /// encodes, decodes, hashes and comparisons of many entries) holding
    /// whatever lock the binding's value code takes per call, so a batch
    /// takes it once instead of once per call. incr-persist calls it under
    /// store and memo locks, and a batch waits for no lock but the
    /// binding's own; `hold` must call its argument exactly once, on the
    /// calling thread. A store also drops the values a call spent inside
    /// `hold`, after releasing its lock; a `Drop` that calls a store from
    /// there must first release whatever `hold` took, as the Python
    /// bindings release the GIL, or it can deadlock against a store call
    /// waiting for that lock.
    pub fn holding(self, hold: fn(&mut dyn FnMut())) -> Self {
        ValueScope {
            hold: Some(hold),
            ..self
        }
    }

    /// This scope with `idle`, the inverse of `hold`: it runs a stretch of
    /// incr-persist's own work that makes no value call having given up
    /// whatever `hold` takes, so other threads can take it meanwhile. A
    /// checkpoint worker runs its pulls inside it while a backend that
    /// holds that lock waits for the next record; a `hold` inside the
    /// stretch takes the lock again for its batch. `idle` must call its
    /// argument exactly once, on the calling thread.
    pub fn idling(self, idle: fn(&mut dyn FnMut())) -> Self {
        ValueScope {
            idle: Some(idle),
            ..self
        }
    }

    /// This scope with `wait`, which runs a blocking wait for a checkpoint
    /// worker on the thread that owns the store or memo, possibly while
    /// incr-persist holds its own locks there, such as a store's lock
    /// across the wait for a worker that is committing. Unlike `idle`, it
    /// must give up whatever the worker may need to finish, whatever thread
    /// it runs on, since the worker waits for that while this thread waits
    /// for the worker; the Python bindings release the GIL. `wait` must
    /// call its argument exactly once, on the calling thread.
    pub fn waiting(self, wait: fn(&mut dyn FnMut())) -> Self {
        ValueScope {
            wait: Some(wait),
            ..self
        }
    }

    /// This scope with `start`, which runs the whole life of a checkpoint
    /// worker thread. incr-persist spawns the thread, makes this scope
    /// current on it and calls `start` once with the worker's body, so a
    /// binding can keep per-thread state for the worker's whole life and run
    /// the body without whatever `hold` takes. `start` must call its
    /// argument exactly once, on the calling thread.
    pub fn starting_workers(self, start: fn(&mut dyn FnMut())) -> Self {
        ValueScope {
            start: Some(start),
            ..self
        }
    }

    /// The state this scope was created with, if it is a `T`.
    pub fn downcast<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        Arc::clone(&self.state).downcast().ok()
    }
}

/// Runs `f`, a batch of value calls that waits for no store or memo lock,
/// inside `scope`'s hold, or directly without one.
pub(crate) fn hold(scope: Option<&ValueScope>, f: impl FnOnce()) {
    run_inside(scope.and_then(|scope| scope.hold), f, "hold");
}

/// Runs `f`, incr-persist's own work with no value call, inside `scope`'s
/// idle, or directly without one.
pub(crate) fn idle(scope: Option<&ValueScope>, f: impl FnOnce()) {
    run_inside(scope.and_then(|scope| scope.idle), f, "idle");
}

/// Runs `f`, a caller's blocking wait for a checkpoint worker, inside
/// `scope`'s wait, or directly without one.
pub(crate) fn wait_released(scope: Option<&ValueScope>, f: impl FnOnce()) {
    run_inside(scope.and_then(|scope| scope.wait), f, "wait");
}

/// Runs `f`, a checkpoint worker's body, inside `scope`'s worker start, or
/// directly without one.
pub(crate) fn start_worker(scope: Option<&ValueScope>, f: impl FnOnce()) {
    run_inside(scope.and_then(|scope| scope.start), f, "worker start");
}

fn run_inside(wrapper: Option<fn(&mut dyn FnMut())>, f: impl FnOnce(), name: &str) {
    let mut f = Some(f);
    let mut run = || {
        if let Some(f) = f.take() {
            f();
        }
    };
    match wrapper {
        Some(wrapper) => wrapper(&mut run),
        None => run(),
    }
    assert!(
        f.is_none(),
        "incr-persist: a value scope's {name} did not run its argument"
    );
}

impl fmt::Debug for ValueScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ValueScope(..)")
    }
}

thread_local! {
    static SCOPE: RefCell<Option<ValueScope>> = const { RefCell::new(None) };
}

/// The value scope entered on this thread, if any.
pub fn current_value_scope() -> Option<ValueScope> {
    SCOPE.with(|scope| scope.borrow().clone())
}

/// Makes `scope` current on this thread until dropped, then restores the
/// previous one, also when unwinding.
pub fn enter_value_scope(scope: ValueScope) -> ValueScopeGuard {
    let previous = SCOPE.with(|current| current.replace(Some(scope)));
    ValueScopeGuard { previous }
}

#[must_use = "the scope ends when the guard is dropped"]
pub struct ValueScopeGuard {
    previous: Option<ValueScope>,
}

impl Drop for ValueScopeGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        let _ = SCOPE.try_with(|current| current.replace(previous));
    }
}

pub(crate) fn in_key<R>(key: &str, f: impl FnOnce() -> R) -> R {
    let _guard = enter_value_context(ValueContext::Key(key.to_owned()));
    f()
}

pub(crate) fn in_memo<R>(table: &str, part: MemoPart, f: impl FnOnce() -> R) -> R {
    let _guard = enter_value_context(ValueContext::MemoTable {
        table: table.to_owned(),
        part,
    });
    f()
}

#[cfg(test)]
pub(crate) mod wait_probe {
    use std::cell::{Cell, RefCell};
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::ValueScope;

    /// How long a worker lets a caller that entered the wait stay there
    /// before it records whether the caller still is.
    const SETTLE: Duration = Duration::from_millis(20);

    /// Whether a caller is inside [`counting_waits`]'s `wait`, and what a
    /// checkpoint worker saw of that when it let itself finish.
    #[derive(Default)]
    pub(crate) struct WaitProbe {
        inside: AtomicBool,
        seen: Mutex<Option<bool>>,
    }

    impl WaitProbe {
        /// On the worker, before it finishes: waits up to `timeout` for the
        /// caller to enter the wait, then records whether it is still inside
        /// [`SETTLE`] later. Only the first call records.
        pub(crate) fn record(&self, timeout: Duration) {
            if self.seen().is_some() {
                return;
            }
            let started = Instant::now();
            while !self.inside.load(SeqCst) && started.elapsed() < timeout {
                thread::sleep(Duration::from_millis(1));
            }
            thread::sleep(SETTLE);
            *self.seen.lock().unwrap() = Some(self.inside.load(SeqCst));
        }

        pub(crate) fn seen(&self) -> Option<bool> {
            *self.seen.lock().unwrap()
        }
    }

    thread_local! {
        static WAITS: Cell<u32> = const { Cell::new(0) };
        static PROBE: RefCell<Option<Arc<WaitProbe>>> = const { RefCell::new(None) };
    }

    fn counted_wait(wait: &mut dyn FnMut()) {
        WAITS.set(WAITS.get() + 1);
        let probe = PROBE.with(|probe| probe.borrow().clone());
        if let Some(probe) = &probe {
            probe.inside.store(true, SeqCst);
        }
        wait();
        if let Some(probe) = &probe {
            probe.inside.store(false, SeqCst);
        }
    }

    /// A value scope whose `wait` counts its entries on the calling thread
    /// and marks the probe [`watch`] set there inside while its argument
    /// runs.
    pub(crate) fn counting_waits() -> ValueScope {
        ValueScope::new(Arc::new(())).waiting(counted_wait)
    }

    /// Resets this thread's wait count and makes `probe` the one its waits
    /// mark.
    pub(crate) fn watch(probe: Option<Arc<WaitProbe>>) {
        WAITS.set(0);
        PROBE.set(probe);
    }

    pub(crate) fn waits() -> u32 {
        WAITS.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str) -> ValueContext {
        ValueContext::Key(name.to_owned())
    }

    #[test]
    fn no_context_outside_a_scope() {
        assert_eq!(current_value_context(), None);
    }

    #[test]
    fn nested_guards_restore_in_order() {
        let outer = enter_value_context(key("a"));
        {
            let _inner = enter_value_context(key("b"));
            assert_eq!(current_value_context(), Some(key("b")));
        }
        assert_eq!(current_value_context(), Some(key("a")));
        drop(outer);
        assert_eq!(current_value_context(), None);
    }

    #[test]
    fn a_panic_restores_the_previous_context() {
        let _outer = enter_value_context(key("a"));
        let result = std::panic::catch_unwind(|| {
            let _inner = enter_value_context(key("b"));
            panic!("unwind");
        });
        assert!(result.is_err());
        assert_eq!(current_value_context(), Some(key("a")));
    }

    #[test]
    fn contexts_are_per_thread() {
        let _guard = enter_value_context(key("a"));
        std::thread::spawn(|| assert_eq!(current_value_context(), None))
            .join()
            .unwrap();
    }
}
