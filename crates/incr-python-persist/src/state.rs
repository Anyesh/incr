//! The binding's own state for one runtime, which every Python callable
//! handed to the engine carries, so code running inside a query reaches it.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;

use crate::binding::PyBinding;
use crate::overlay::{Born, Overlay};

/// One runtime's binding state: how many Python callables its engine holds
/// (one per live [`Held`] made from it), the active overlay's journal, and
/// the package's own per-runtime state `X`.
pub struct RtState<B: PyBinding, X = ()>(Arc<Inner<B, X>>);

impl<B: PyBinding, X> Clone for RtState<B, X> {
    fn clone(&self) -> Self {
        RtState(Arc::clone(&self.0))
    }
}

impl<B: PyBinding, X: Default> Default for RtState<B, X> {
    fn default() -> Self {
        RtState(Arc::new(Inner {
            callables: AtomicUsize::new(0),
            active: AtomicBool::new(false),
            overlay: Mutex::new(Journal::Idle),
            extra: X::default(),
        }))
    }
}

struct Inner<B: PyBinding, X> {
    callables: AtomicUsize,
    active: AtomicBool,
    /// A leaf lock: the journal is lent out of it while code that may run
    /// Python works on it.
    overlay: Mutex<Journal<B>>,
    extra: X,
}

enum Journal<B: PyBinding> {
    Idle,
    Open(Box<Overlay<B>>),
    Lent,
}

/// Puts a lent journal back, also when the code using it unwinds.
struct Lend<'a, B: PyBinding> {
    slot: &'a Mutex<Journal<B>>,
    overlay: Option<Box<Overlay<B>>>,
}

impl<B: PyBinding> Drop for Lend<'_, B> {
    fn drop(&mut self) {
        if let Some(overlay) = self.overlay.take() {
            let mut slot = lock(self.slot);
            if matches!(*slot, Journal::Lent) {
                *slot = Journal::Open(overlay);
            }
        }
    }
}

fn lock<B: PyBinding>(slot: &Mutex<Journal<B>>) -> MutexGuard<'_, Journal<B>> {
    slot.lock().unwrap_or_else(PoisonError::into_inner)
}

impl<B: PyBinding, X> RtState<B, X> {
    /// `f` counted as one of this runtime's callables until the returned
    /// handle drops.
    pub fn hold(&self, f: Py<PyAny>) -> Held<B, X> {
        self.0.callables.fetch_add(1, Ordering::Relaxed);
        Held {
            f,
            state: self.clone(),
        }
    }

    pub fn callables(&self) -> usize {
        self.0.callables.load(Ordering::Relaxed)
    }

    /// The package's own per-runtime state.
    pub fn extra(&self) -> &X {
        &self.0.extra
    }

    pub fn overlay_active(&self) -> bool {
        self.0.active.load(Ordering::Acquire)
    }

    /// Runs `f` on the active overlay's journal, or does nothing outside
    /// one.
    pub fn journal(&self, f: impl FnOnce(&mut Overlay<B>)) {
        self.journal_with(f);
    }

    /// [`Self::journal`] returning what `f` returns, `None` outside an
    /// overlay. `f` runs with the journal lent out of the lock.
    ///
    /// # Panics
    ///
    /// When `f` reaches the journal again, as a nested borrow would.
    pub fn journal_with<R>(&self, f: impl FnOnce(&mut Overlay<B>) -> R) -> Option<R> {
        if !self.overlay_active() {
            return None;
        }
        let overlay = {
            let mut slot = lock(&self.0.overlay);
            match std::mem::replace(&mut *slot, Journal::Lent) {
                Journal::Open(overlay) => overlay,
                Journal::Idle => {
                    *slot = Journal::Idle;
                    return None;
                }
                Journal::Lent => panic!("the overlay journal is already in use"),
            }
        };
        let mut lend = Lend {
            slot: &self.0.overlay,
            overlay: Some(overlay),
        };
        Some(f(lend.overlay.as_mut().expect("lent until dropped")))
    }

    /// The marker for an object created now: dead when the active
    /// overlay ends, never outside one.
    pub fn born(&self) -> Born {
        self.journal_with(Overlay::born).unwrap_or_default()
    }

    pub fn begin_overlay(&self, overlay: Overlay<B>) {
        *lock(&self.0.overlay) = Journal::Open(Box::new(overlay));
        self.0.active.store(true, Ordering::Release);
    }

    pub fn end_overlay(&self) -> Option<Overlay<B>> {
        self.0.active.store(false, Ordering::Release);
        match std::mem::replace(&mut *lock(&self.0.overlay), Journal::Idle) {
            Journal::Open(overlay) => Some(*overlay),
            Journal::Idle | Journal::Lent => None,
        }
    }
}

/// A Python callable handed to the engine, counted in its runtime's
/// [`RtState`] from creation until drop. It is `Send` and `Sync` exactly
/// when the state is; a package whose state is neither wraps it under its
/// own confinement argument.
pub struct Held<B: PyBinding, X = ()> {
    f: Py<PyAny>,
    state: RtState<B, X>,
}

impl<B: PyBinding, X> Held<B, X> {
    pub fn state(&self) -> &RtState<B, X> {
        &self.state
    }
}

impl<B: PyBinding, X> Drop for Held<B, X> {
    fn drop(&mut self) {
        self.state.0.callables.fetch_sub(1, Ordering::Relaxed);
    }
}

impl<B: PyBinding, X> Deref for Held<B, X> {
    type Target = Py<PyAny>;

    fn deref(&self) -> &Py<PyAny> {
        &self.f
    }
}
