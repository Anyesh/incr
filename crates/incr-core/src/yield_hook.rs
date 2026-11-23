//! A runtime's yield hook: a callback the binding installs so that long
//! native work lets other threads of its host run, as the Python binding
//! hands over the GIL. The engine calls it at each compute and every
//! [`YIELD_EVERY`] iterations of its long loops (walks, dirty marking,
//! batch applies, drains, and incr-persist's restores and checkpoints).
//!
//! It calls the hook only where no other thread can be waiting on the
//! caller: where user code could run as well (a compute boundary, a walk
//! step between computes), or, on a runtime whose cells are
//! single-threaded and so entered by no other thread, under its locks and
//! inside work that runs no user code. No call site holds a process-wide
//! lock (the wait registry, the reservation table or the hold sets), which
//! runtimes on other threads share.

use std::cell::Cell;
#[cfg(feature = "yield-hook")]
use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

use crate::cells::Cells;
use crate::runtime::Runtime;

/// Iterations of a long loop between two calls of the hook.
pub const YIELD_EVERY: u32 = 1024;

/// The hook must neither unwind nor call into any runtime: it can run in
/// the middle of a batch whose state no user code sees.
#[derive(Clone)]
pub struct YieldHook(Arc<dyn Fn() + Send + Sync>);

impl YieldHook {
    pub fn new(hook: impl Fn() + Send + Sync + 'static) -> Self {
        YieldHook(Arc::new(hook))
    }

    #[cold]
    #[inline(never)]
    fn call(&self) {
        (self.0)()
    }
}

impl fmt::Debug for YieldHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("YieldHook")
    }
}

#[cfg(feature = "yield-hook")]
thread_local! {
    static AMBIENT: RefCell<Option<YieldHook>> = const { RefCell::new(None) };
}

/// Runs `work` with `hook` as this thread's ambient hook, the one
/// [`Yielder::ambient`] ticks: for code a runtime's work calls that has no
/// runtime to ask, such as a persistence backend. Every tick site under it
/// follows the rules of the runtime's own sites; a site that holds a lock
/// of its own, one another thread could wait on while it keeps the host
/// from running, must not tick.
#[cfg(feature = "yield-hook")]
pub fn with_ambient_hook<R>(hook: Option<&YieldHook>, work: impl FnOnce() -> R) -> R {
    struct Restore(Option<YieldHook>);
    impl Drop for Restore {
        fn drop(&mut self) {
            AMBIENT.with(|ambient| *ambient.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(AMBIENT.with(|ambient| ambient.replace(hook.cloned())));
    work()
}

enum HookRef<'a> {
    Borrowed(&'a YieldHook),
    #[cfg(feature = "yield-hook")]
    Owned(YieldHook),
}

impl HookRef<'_> {
    #[cold]
    #[inline(never)]
    fn call(&self) {
        match self {
            HookRef::Borrowed(hook) => hook.call(),
            #[cfg(feature = "yield-hook")]
            HookRef::Owned(hook) => hook.call(),
        }
    }
}

/// Calls a runtime's hook every [`YIELD_EVERY`] ticks; with no hook, a
/// tick is one branch on a value the loop keeps in a register.
pub struct Yielder<'a> {
    hook: Option<HookRef<'a>>,
    left: Cell<u32>,
}

impl<'a> Yielder<'a> {
    pub const fn none() -> Self {
        Yielder::of(None)
    }

    /// A [`Yielder`] of `hook` for work outside a runtime, such as a
    /// backend's, under the same rules as the runtime's own.
    pub const fn of(hook: Option<&'a YieldHook>) -> Self {
        Yielder {
            hook: match hook {
                Some(hook) => Some(HookRef::Borrowed(hook)),
                None => None,
            },
            left: Cell::new(YIELD_EVERY),
        }
    }

    /// A [`Yielder`] of this thread's ambient hook (`with_ambient_hook`);
    /// one that never ticks without the `yield-hook` feature.
    #[inline]
    pub fn ambient() -> Yielder<'static> {
        #[cfg(feature = "yield-hook")]
        let hook = AMBIENT.with(|ambient| ambient.borrow().clone().map(HookRef::Owned));
        #[cfg(not(feature = "yield-hook"))]
        let hook = None;
        Yielder {
            hook,
            left: Cell::new(YIELD_EVERY),
        }
    }

    #[inline(always)]
    pub fn tick(&self) {
        if let Some(hook) = &self.hook {
            self.count_down(hook, 1);
        }
    }

    pub fn is_active(&self) -> bool {
        self.hook.is_some()
    }

    /// Collects `iter` with a tick per item it yields, so a filtering
    /// caller ticks its source instead. Without a hook it is a plain
    /// `collect`, which keeps an exact-size source's single allocation.
    #[inline]
    pub fn collect<I: Iterator, B: FromIterator<I::Item>>(&self, iter: I) -> B {
        if self.hook.is_none() {
            return iter.collect();
        }
        iter.inspect(|_| self.tick()).collect()
    }

    /// `n` ticks at once, for a loop that steps a batch at a time.
    #[inline(always)]
    pub fn tick_by(&self, n: usize) {
        if let Some(hook) = &self.hook {
            self.count_down(hook, u32::try_from(n).unwrap_or(u32::MAX));
        }
    }

    #[inline(always)]
    fn count_down(&self, hook: &HookRef<'_>, n: u32) {
        match self.left.get().checked_sub(n) {
            Some(left) if left > 0 => self.left.set(left),
            _ => {
                self.left.set(YIELD_EVERY);
                hook.call();
            }
        }
    }
}

/// What a long loop ticks: a [`Yielder`], or [`NoTicks`] where no hook is
/// set, so that a loop built for it carries no tick at all.
pub trait Ticks {
    fn tick(&self);
    /// Calls the hook now, as at a compute.
    fn yield_now(&self);
    fn is_active(&self) -> bool;
    fn collect<I: Iterator, B: FromIterator<I::Item>>(&self, iter: I) -> B;
}

#[derive(Clone, Copy)]
pub struct NoTicks;

/// Evaluates `$body` with `$ticks` bound to a [`Yielder`] of `$hook`, an
/// `Option<&YieldHook>`, or to [`NoTicks`] when it is `None`; `$body` is
/// built once for each.
#[cfg(not(feature = "yield-hook"))]
macro_rules! with_ticks {
    ($hook:expr, |$ticks:ident| $body:expr) => {{
        let _: Option<&$crate::yield_hook::YieldHook> = $hook;
        let $ticks = &$crate::yield_hook::NoTicks;
        $body
    }};
}
#[cfg(feature = "yield-hook")]
macro_rules! with_ticks {
    ($hook:expr, |$ticks:ident| $body:expr) => {
        match $hook {
            Some(hook) => {
                let $ticks = &$crate::yield_hook::Yielder::of(Some(hook));
                $body
            }
            None => {
                let $ticks = &$crate::yield_hook::NoTicks;
                $body
            }
        }
    };
}
pub(crate) use with_ticks;

impl Ticks for NoTicks {
    #[inline(always)]
    fn tick(&self) {}

    #[inline(always)]
    fn yield_now(&self) {}

    #[inline(always)]
    fn is_active(&self) -> bool {
        false
    }

    #[inline(always)]
    fn collect<I: Iterator, B: FromIterator<I::Item>>(&self, iter: I) -> B {
        iter.collect()
    }
}

impl<T: Ticks> Ticks for &T {
    #[inline(always)]
    fn tick(&self) {
        T::tick(self)
    }

    #[inline(always)]
    fn yield_now(&self) {
        T::yield_now(self)
    }

    #[inline(always)]
    fn is_active(&self) -> bool {
        T::is_active(self)
    }

    #[inline(always)]
    fn collect<I: Iterator, B: FromIterator<I::Item>>(&self, iter: I) -> B {
        T::collect(self, iter)
    }
}

impl Ticks for Yielder<'_> {
    #[inline(always)]
    fn tick(&self) {
        Yielder::tick(self)
    }

    #[inline(always)]
    fn yield_now(&self) {
        if let Some(hook) = &self.hook {
            hook.call();
        }
    }

    #[inline(always)]
    fn is_active(&self) -> bool {
        Yielder::is_active(self)
    }

    #[inline(always)]
    fn collect<I: Iterator, B: FromIterator<I::Item>>(&self, iter: I) -> B {
        Yielder::collect(self, iter)
    }
}

#[cfg(feature = "yield-hook")]
impl<C: Cells> Runtime<C> {
    /// Installs `hook`, or removes the hook with `None`.
    pub fn set_yield_hook(&mut self, hook: Option<YieldHook>) {
        self.yield_hook = hook;
    }

    pub fn yield_hook(&self) -> Option<&YieldHook> {
        self.yield_hook.as_ref()
    }
}

impl<C: Cells> Runtime<C> {
    #[inline(always)]
    pub(crate) fn hook(&self) -> Option<&YieldHook> {
        #[cfg(feature = "yield-hook")]
        return self.yield_hook.as_ref();
        #[cfg(not(feature = "yield-hook"))]
        return None;
    }

    #[doc(hidden)]
    /// Calls the hook, if one is installed. For a caller where user code
    /// could run.
    #[inline(always)]
    pub fn maybe_yield(&self) {
        if let Some(hook) = self.hook() {
            hook.call();
        }
    }

    #[doc(hidden)]
    /// A [`Yielder`] for a loop where user code could run.
    #[inline(always)]
    pub fn yielder(&self) -> Yielder<'_> {
        Yielder::of(self.hook())
    }

    #[doc(hidden)]
    /// Runs `work`, code with no runtime to ask, with this runtime's hook as
    /// the ambient one when the cells are single-threaded, and with none
    /// otherwise, as [`Self::yielder_under_lock`] ticks.
    pub fn with_ambient_under_lock<R>(&self, work: impl FnOnce() -> R) -> R {
        #[cfg(feature = "yield-hook")]
        return with_ambient_hook(self.hook_under_lock(), work);
        #[cfg(not(feature = "yield-hook"))]
        return work();
    }

    /// The hook [`Self::yielder_under_lock`] ticks, if any.
    #[inline(always)]
    pub(crate) fn hook_under_lock(&self) -> Option<&YieldHook> {
        self.hook().filter(|_| C::SINGLE_THREADED)
    }

    #[doc(hidden)]
    /// A [`Yielder`] for a loop under a lock of this runtime, or of a store
    /// or collection on it, or one that other threads wait on: it ticks
    /// only when the cells are single-threaded.
    #[inline(always)]
    pub fn yielder_under_lock(&self) -> Yielder<'_> {
        if C::SINGLE_THREADED {
            self.yielder()
        } else {
            Yielder::none()
        }
    }
}
