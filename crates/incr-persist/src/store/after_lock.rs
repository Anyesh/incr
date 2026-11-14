use std::any::Any;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::value_context::hold;
use crate::ValueScope;

/// User values a store call is done with while it holds the store mutex:
/// displaced values, decoded copies, clones. They are dropped once the
/// mutex is released, so a `Drop` that calls the store does not run under
/// its lock.
#[derive(Default)]
pub(super) struct AfterLock {
    values: RefCell<Vec<Box<dyn Any + Send>>>,
    /// One accumulator per type, see [`AfterLock::with`].
    accumulators: RefCell<Vec<Box<dyn Any + Send>>>,
}

impl AfterLock {
    pub(super) fn push<T: Send + 'static>(&self, value: T) {
        if std::mem::needs_drop::<T>() {
            self.values.borrow_mut().push(Box::new(value));
        }
    }

    /// Runs `f` on the call's one `T`, created on first use, so that the
    /// many writes of one call append their spent values to the same
    /// buffers instead of each keeping its own allocations alive until
    /// the lock is released. `f` must not use this bag.
    pub(super) fn with<T: Default + Send + 'static, R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut accumulators = self.accumulators.borrow_mut();
        let at = match accumulators.iter().position(|kept| (**kept).is::<T>()) {
            Some(at) => at,
            None => {
                accumulators.push(Box::new(T::default()));
                accumulators.len() - 1
            }
        };
        f(accumulators[at]
            .downcast_mut::<T>()
            .expect("an accumulator is found by its type"))
    }

    /// [`Self::push`], keeping nothing for an empty `values`, so a call that
    /// spent nothing takes no hold after its lock.
    pub(super) fn push_vec<T: Send + 'static>(&self, values: Vec<T>) {
        if !values.is_empty() {
            self.push(values);
        }
    }
}

/// The values taken from an [`AfterLock`] as the mutex is released,
/// dropped when this is, inside `scope`'s hold so that a binding's value
/// code (Python decrefs on `Shared`) runs attached and batched. An empty
/// bag takes no hold.
pub(super) struct Drained<'a> {
    values: Vec<Box<dyn Any + Send>>,
    scope: Option<&'a ValueScope>,
}

impl<'a> Drained<'a> {
    pub(super) fn new(scope: Option<&'a ValueScope>) -> Self {
        Drained {
            values: Vec::new(),
            scope,
        }
    }

    pub(super) fn take_from(&mut self, bag: &AfterLock) {
        self.values.append(&mut bag.values.borrow_mut());
        self.values.append(&mut bag.accumulators.borrow_mut());
    }
}

impl Drop for Drained<'_> {
    fn drop(&mut self) {
        if self.values.is_empty() {
            return;
        }
        let values = std::mem::take(&mut self.values);
        hold(self.scope, || drop(values));
    }
}

thread_local! {
    static TOKEN: Cell<u8> = const { Cell::new(0) };
}

/// A nonzero value unique to the calling thread among live threads.
fn thread_token() -> usize {
    TOKEN.with(|token| token as *const Cell<u8> as usize)
}

/// The thread holding a store's mutex, if any, so that a call back into
/// the store on that thread fails instead of deadlocking.
#[derive(Default)]
pub(super) struct Holder {
    thread: AtomicUsize,
    poison: AtomicBool,
}

impl Holder {
    /// Whether the calling thread holds the mutex. Only the holding thread
    /// stores its own token, so a relaxed load sees it exactly when this
    /// thread set it and has not cleared it.
    pub(super) fn is_current(&self) -> bool {
        self.thread.load(Ordering::Relaxed) == thread_token()
    }

    /// Marks the calling thread as the holder until the mark drops.
    pub(super) fn mark(&self) -> HolderMark<'_> {
        self.thread.store(thread_token(), Ordering::Relaxed);
        HolderMark(self)
    }

    /// Asks the holding call, which must be the caller's own, to poison the
    /// store before it releases the mutex.
    pub(super) fn request_poison(&self) {
        self.poison.store(true, Ordering::Relaxed);
    }
}

pub(super) struct HolderMark<'a>(&'a Holder);

impl HolderMark<'_> {
    /// Whether a poison was requested during this hold; clears the request.
    pub(super) fn take_poison(&self) -> bool {
        self.0.poison.swap(false, Ordering::Relaxed)
    }
}

impl Drop for HolderMark<'_> {
    fn drop(&mut self) {
        self.0.thread.store(0, Ordering::Relaxed);
    }
}
