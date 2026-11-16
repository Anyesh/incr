//! A Rust iterator lent to a Python backend for the length of one call:
//! the shared core of the lazy `frames` and `records` iterators.

use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, TryLockError};
use std::thread::{self, ThreadId};

use incr_persist::{Error, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(crate) type Pulls<'a, T> = dyn Iterator<Item = Result<T>> + 'a;

struct Source<T: 'static>(*mut Pulls<'static, T>);

// The pointer is only dereferenced on the thread that lent it (`pull`
// checks), while the lender keeps its referent alive.
unsafe impl<T> Send for Source<T> {}

/// What stopped a pull: an error the lent iterator returned, or a panic
/// inside it, resumed once the backend's call has returned.
pub(crate) enum Failure {
    Error(Error),
    Panic(Box<dyn std::any::Any + Send>),
}

/// How a lent iterator names itself in the errors a Python caller sees.
pub(crate) struct Names {
    pub items: &'static str,
    pub call: &'static str,
    pub source: &'static str,
}

pub(crate) struct Lending<T: 'static> {
    source: Mutex<Option<Source<T>>>,
    thread: ThreadId,
    exhausted: AtomicBool,
    failure: Mutex<Option<Failure>>,
    names: Names,
}

impl<T> Lending<T> {
    /// Lends `pulls` until [`Lending::invalidate`]; the caller must call
    /// that before `pulls` is dropped or used again.
    pub(crate) fn new(pulls: &mut Pulls<'_, T>, names: Names) -> Self {
        let ptr: *mut Pulls<'_, T> = pulls;
        // SAFETY: erases the borrow's lifetime; `pull` never uses the
        // pointer once the lender has invalidated it.
        let ptr = unsafe { std::mem::transmute::<*mut Pulls<'_, T>, *mut Pulls<'static, T>>(ptr) };
        Lending {
            source: Mutex::new(Some(Source(ptr))),
            thread: thread::current().id(),
            exhausted: AtomicBool::new(false),
            failure: Mutex::new(None),
            names,
        }
    }

    pub(crate) fn exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Acquire)
    }

    pub(crate) fn failure(&self) -> Option<Failure> {
        self.failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }

    pub(crate) fn invalidate(&self) {
        *self.source.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    pub(crate) fn pull(&self) -> PyResult<Option<T>> {
        let Names {
            items,
            call,
            source: name,
        } = self.names;
        if thread::current().id() != self.thread {
            return Err(PyRuntimeError::new_err(format!(
                "{items} must be pulled on the thread that called {call}"
            )));
        }
        let mut source = match self.source.try_lock() {
            Ok(source) => source,
            Err(TryLockError::WouldBlock) => {
                return Err(PyRuntimeError::new_err(format!(
                    "{items} pulled re-entrantly"
                )))
            }
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
        };
        let Some(Source(pulls)) = source.as_mut() else {
            return Err(PyRuntimeError::new_err(format!(
                "{items} are only valid during the {call} call"
            )));
        };
        if self.exhausted() {
            return Ok(None);
        }
        // SAFETY: the lender keeps the pointee alive and clears this slot
        // before the call that lent it returns, and the lock excludes a
        // second pull.
        let next = panic::catch_unwind(AssertUnwindSafe(|| unsafe { (**pulls).next() }));
        let next = match next {
            Ok(next) => next,
            Err(payload) => {
                *source = None;
                *self.failure.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(Failure::Panic(payload));
                return Err(PyRuntimeError::new_err(format!("{name} panicked")));
            }
        };
        match next {
            None => {
                self.exhausted.store(true, Ordering::Release);
                Ok(None)
            }
            Some(Ok(item)) => Ok(Some(item)),
            Some(Err(error)) => Err(self.fail(error)),
        }
    }

    /// Records `error` as what stopped the pulls, for the lender to return
    /// after the call, and gives the exception the puller sees.
    pub(crate) fn fail(&self, error: Error) -> PyErr {
        let message = format!("{} failed: {error}", self.names.source);
        *self.failure.lock().unwrap_or_else(|e| e.into_inner()) = Some(Failure::Error(error));
        PyRuntimeError::new_err(message)
    }
}
