//! [`Confined`]: a value kept on the thread that made it, for the Local
//! binding's pyclasses.
//!
//! PyO3's `unsendable` checks the thread on access too, but when Python
//! frees such an object on another thread (a garbage collection there, or
//! the last reference held by something that thread drops) it skips the
//! Rust drop and reports an unraisable error. A `Confined` value freed off
//! its thread is instead queued for that thread, which drops it at its
//! next engine call; the main thread also drops its own queue from a
//! pending call. Once the owner thread has exited, its queue and any later
//! frees go to the main thread, which then is the only thread that can
//! reach them.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::{ffi, Python};

/// `T` usable only on the thread that made it, and dropped there.
pub struct Confined<T: 'static> {
    owner: u64,
    value: ManuallyDrop<T>,
}

// SAFETY: `Deref` and `DerefMut` refuse every thread but `owner`, and a
// value freed on another thread is moved, never touched, to the owner's
// queue and dropped by the owner (or, once the owner has exited, by the
// main thread alone).
unsafe impl<T> Send for Confined<T> {}
// SAFETY: see `Send`.
unsafe impl<T> Sync for Confined<T> {}

impl<T: 'static> Confined<T> {
    pub fn new(value: T) -> Self {
        Confined {
            owner: owner_id(),
            value: ManuallyDrop::new(value),
        }
    }

    #[inline]
    fn check(&self) {
        if ID.with(Cell::get) != self.owner {
            wrong_thread::<T>()
        }
    }
}

#[cold]
#[inline(never)]
fn wrong_thread<T>() -> ! {
    let fields = std::any::type_name::<T>();
    let class = fields.strip_suffix("Fields").unwrap_or(fields);
    panic!("{class} is unsendable, but sent to another thread")
}

impl<T> Deref for Confined<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.check();
        &self.value
    }
}

impl<T> DerefMut for Confined<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.check();
        &mut self.value
    }
}

impl<T: Clone> Clone for Confined<T> {
    fn clone(&self) -> Self {
        Confined {
            owner: self.owner,
            value: ManuallyDrop::new(T::clone(self)),
        }
    }
}

impl<T> Drop for Confined<T> {
    fn drop(&mut self) {
        // SAFETY: `value` is taken here only, and never used after.
        let value = unsafe { ManuallyDrop::take(&mut self.value) };
        if ID.with(Cell::get) == self.owner {
            drop(value);
        } else {
            defer(self.owner, Deferred(Box::new(value)));
        }
    }
}

/// A shared newtype with the same shape as [`Confined`], for a binding
/// whose classes any thread may use.
#[repr(transparent)]
pub struct Unconfined<T>(T);

impl<T> Deref for Unconfined<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Unconfined<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: 'static> From<T> for Confined<T> {
    fn from(value: T) -> Self {
        Confined::new(value)
    }
}

impl<T> From<T> for Unconfined<T> {
    #[inline]
    fn from(value: T) -> Self {
        Unconfined(value)
    }
}

/// Makes the pyclass newtype `$class`, a tuple struct over a [`Confined`]
/// or [`Unconfined`] of `$fields`, deref to `$fields` and build from it,
/// so its methods read the fields as their own.
#[macro_export]
macro_rules! confined_newtype {
    ($class:ident => $fields:ty) => {
        impl ::std::ops::Deref for $class {
            type Target = $fields;

            #[inline]
            fn deref(&self) -> &$fields {
                &self.0
            }
        }

        impl ::std::ops::DerefMut for $class {
            #[inline]
            fn deref_mut(&mut self) -> &mut $fields {
                &mut self.0
            }
        }

        impl ::std::convert::From<$fields> for $class {
            #[inline]
            fn from(fields: $fields) -> Self {
                $class(::std::convert::From::from(fields))
            }
        }
    };
}

trait Erased {}

impl<T> Erased for T {}

/// A value freed off its owner thread, waiting to be dropped.
struct Deferred(#[expect(dead_code, reason = "held only to be dropped")] Box<dyn Erased>);

// SAFETY: a `Deferred` is moved between threads but dropped only on its
// owner thread, or on the main thread once the owner has exited.
unsafe impl Send for Deferred {}

struct Queue {
    pending: Arc<AtomicBool>,
    items: Vec<Deferred>,
}

struct Registry {
    live: BTreeMap<u64, Queue>,
    orphans: Vec<Deferred>,
}

impl Registry {
    fn any_queued(&self) -> bool {
        !self.orphans.is_empty() || self.live.values().any(|q| !q.items.is_empty())
    }
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    live: BTreeMap::new(),
    orphans: Vec::new(),
});

/// Set while any queue may hold a value; the one check on the hot path.
static PENDING: AtomicBool = AtomicBool::new(false);
static SCHEDULED: AtomicBool = AtomicBool::new(false);
/// Set while the orphans may be non-empty, so a failed `schedule` is
/// retried.
static ORPHANED: AtomicBool = AtomicBool::new(false);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// This thread's owner id, 0 until it makes its first `Confined`.
    static ID: Cell<u64> = const { Cell::new(0) };
    static HOME: RefCell<Option<Home>> = const { RefCell::new(None) };
}

/// A registered thread's queue flag; dropping it at thread exit hands
/// the thread's queue to the main thread.
struct Home {
    id: u64,
    pending: Arc<AtomicBool>,
}

impl Drop for Home {
    fn drop(&mut self) {
        let mut registry = registry();
        if let Some(queue) = registry.live.remove(&self.id) {
            if !queue.items.is_empty() {
                registry.orphans.extend(queue.items);
                // Not scheduled from here: the interpreter may be finalizing
                // once `join` has returned. The next free or engine call
                // schedules the main thread's drain.
                ORPHANED.store(true, Ordering::Release);
                PENDING.store(true, Ordering::Release);
            }
        }
    }
}

fn registry() -> MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)
}

fn owner_id() -> u64 {
    match ID.with(Cell::get) {
        0 => register(),
        id => id,
    }
}

#[cold]
#[inline(never)]
fn register() -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let pending = Arc::new(AtomicBool::new(false));
    let home = Home {
        id,
        pending: Arc::clone(&pending),
    };
    // A thread past its thread-local destructors cannot hold a `Home`, so
    // it stays out of `live` and anything freed elsewhere goes to the main
    // thread.
    if HOME.try_with(|h| *h.borrow_mut() = Some(home)).is_ok() {
        registry().live.insert(
            id,
            Queue {
                pending,
                items: Vec::new(),
            },
        );
    }
    ID.with(|c| c.set(id));
    id
}

fn defer(owner: u64, value: Deferred) {
    let mut registry = registry();
    match registry.live.get_mut(&owner) {
        Some(queue) => {
            queue.items.push(value);
            queue.pending.store(true, Ordering::Release);
        }
        None => {
            registry.orphans.push(value);
            ORPHANED.store(true, Ordering::Release);
        }
    }
    PENDING.store(true, Ordering::Release);
    drop(registry);
    schedule();
}

/// Asks the main thread to drop its queue and the orphans at its next
/// bytecode boundary. Needs no GIL, since it runs from `tp_dealloc` on any
/// thread.
fn schedule() {
    if SCHEDULED.swap(true, Ordering::AcqRel) {
        return;
    }
    // SAFETY: both calls need neither the GIL nor a thread state.
    let added = unsafe {
        ffi::Py_IsInitialized() != 0
            && ffi::Py_AddPendingCall(Some(run_pending), std::ptr::null_mut()) == 0
    };
    if !added {
        SCHEDULED.store(false, Ordering::Release);
    }
}

extern "C" fn run_pending(_: *mut c_void) -> c_int {
    SCHEDULED.store(false, Ordering::Release);
    // Attached, so the `Py` references the drops release are decref'd now
    // rather than pooled until the next PyO3 call.
    Python::attach(|py| {
        if let Err(payload) = std::panic::catch_unwind(|| drain(true)) {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("a value's drop panicked");
            PyRuntimeError::new_err(message.to_owned()).write_unraisable(py, None);
        }
    });
    0
}

/// Drops the values other threads freed for this one. Call with the GIL
/// held, at a point where Python code could run, such as the start of an
/// engine call.
#[inline]
pub fn drain_pending() {
    if PENDING.load(Ordering::Relaxed) {
        drain(false);
    }
}

#[cold]
#[inline(never)]
fn drain(main: bool) {
    let own = HOME
        .try_with(|h| {
            h.borrow().as_ref().is_some_and(|home| {
                home.pending.load(Ordering::Relaxed) && home.pending.swap(false, Ordering::Acquire)
            })
        })
        .unwrap_or(false);
    if !own && !main {
        if ORPHANED.load(Ordering::Relaxed) {
            schedule();
        }
        return;
    }
    let taken = {
        let mut registry = registry();
        let mut taken = Vec::new();
        if own {
            if let Some(queue) = registry.live.get_mut(&ID.with(Cell::get)) {
                taken.append(&mut queue.items);
            }
        }
        if main {
            taken.append(&mut registry.orphans);
            ORPHANED.store(false, Ordering::Release);
        }
        PENDING.store(registry.any_queued(), Ordering::Release);
        taken
    };
    drop(taken);
}
