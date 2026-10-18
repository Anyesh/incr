//! Cycles as values.
//!
//! A query created with [`Runtime::create_query_with`](crate::Runtime::create_query_with)
//! can carry a [`CycleStrategy`]: when a read closes a dependency cycle
//! through it, the cycle resolves to values instead of panicking. A
//! `Fallback` member lands the value its function gives for the
//! [`Cycle`]; a `Fixpoint` member iterates from an initial value until
//! its values converge or an iteration limit is reached. Members with no
//! strategy land the values computed against the strategy members',
//! and those that still read each other land
//! [`NodeState::Cyclic`](crate::NodeState::Cyclic) with no value, which
//! [`Runtime::try_get`](crate::Runtime::try_get) reports as `Err(Cycle)`
//! and [`Runtime::get`](crate::Runtime::get) as a "dependency cycle"
//! panic.
//!
//! The callbacks (`Fallback`'s function, and a `Fixpoint`'s `initial`,
//! `step`, `converged` and `on_limit`) can run more than once per
//! landing. They run with no engine lock held and as value code: a read
//! of a Clean node returns its value and records nothing, and any other
//! call into the runtime panics.
//!
//! # Entry order
//!
//! The landed values are the same whichever member is read first, under
//! `Local` and `Shared`, except in two cases:
//! - a `get` that enters a cycle at a member with no strategy panics
//!   while the cycle is being discovered, because that member has no
//!   value to give the read that closes the cycle; a `try_get` there
//!   lands the same values as any other entry;
//! - which nodes are members can differ when a member's reads depend on
//!   the provisional values the cycle passes around.
//!
//! Under `Shared`, two threads can enter one cycle at once and wait on
//! each other's claims. The wait that closes such a waits-for cycle
//! makes every thread on it but the oldest release its claims and retry
//! its outermost engine call, so the oldest resolves the cycle alone.
//!
//! # `panic = abort`
//!
//! Cycles whose members all have a strategy resolve without unwinding.
//! These outcomes unwind, and so abort when panics abort (as on wasm by
//! default): a `get` that closes a cycle with no value, such a read
//! nested under a `try_get` that would return it as `Err`, and the
//! retry of a `Shared` thread waiting on a claim cycle.

use std::any::Any;
use std::cell::Cell;
use std::fmt;
use std::sync::Arc;

use crate::handle::ErasedIncr;
use crate::value::Value;

/// The members of a dependency cycle, sorted by slot. The head is the
/// lowest-slot participant.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Cycle {
    participants: Arc<[ErasedIncr]>,
}

impl Cycle {
    /// Builds a cycle from its members in any order.
    pub(crate) fn new(mut participants: Vec<ErasedIncr>) -> Self {
        participants.sort_unstable_by_key(|p| p.slot());
        participants.dedup();
        debug_assert!(!participants.is_empty(), "a cycle has a participant");
        Self {
            participants: participants.into(),
        }
    }

    /// The participant with the lowest slot.
    pub fn head(&self) -> ErasedIncr {
        self.participants[0]
    }

    /// Every participant, sorted by slot.
    pub fn participants(&self) -> &[ErasedIncr] {
        &self.participants
    }

    /// The number of participants.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.participants.len()
    }

    /// Whether `node` takes part in the cycle.
    pub fn contains(&self, node: impl Into<ErasedIncr>) -> bool {
        let node = node.into();
        self.participants
            .binary_search_by_key(&node.slot(), |p| p.slot())
            .is_ok_and(|i| self.participants[i] == node)
    }
}

impl fmt::Debug for Cycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.participants.iter().map(|p| p.slot()))
            .finish()
    }
}

impl fmt::Display for Cycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, p) in self.participants.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", p.slot())?;
        }
        Ok(())
    }
}

impl std::error::Error for Cycle {}

type FallbackFn<T> = Box<dyn Fn(&Cycle) -> T + Send + Sync>;
type ConvergedFn<T> = Box<dyn Fn(&T, &T) -> bool + Send + Sync>;
type StepFn<T> = Box<dyn Fn(&T, T, u32) -> T + Send + Sync>;
type OnLimitFn<T> = Box<dyn Fn(&Cycle, T) -> T + Send + Sync>;

/// How a query resolves a dependency cycle through it.
#[derive(Default)]
pub enum CycleStrategy<T> {
    /// No resolution: a read that closes a cycle through this query
    /// panics under `get` and returns `Err(Cycle)` under `try_get`, and a
    /// member that still reads another such member lands `Cyclic`.
    #[default]
    None,
    /// The member's value in a cycle is this function of the cycle.
    Fallback(FallbackFn<T>),
    /// The member iterates from an initial value to a fixed point.
    Fixpoint(Fixpoint<T>),
}

impl<T> fmt::Debug for CycleStrategy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CycleStrategy::None => "None",
            CycleStrategy::Fallback(_) => "Fallback",
            CycleStrategy::Fixpoint(_) => "Fixpoint",
        })
    }
}

impl<T> CycleStrategy<T> {
    /// [`CycleStrategy::Fallback`] from a function.
    pub fn fallback(g: impl Fn(&Cycle) -> T + Send + Sync + 'static) -> Self {
        Self::Fallback(Box::new(g))
    }
}

/// A fixed-point iteration: every member of the cycle starts at
/// `initial()`, and iteration `k` gives each member
/// `step(&previous, output, k)`, where `output` is what its closure
/// returned reading the previous values. The iteration stops when every
/// member has `converged(&previous, &current)`, or after
/// `max_iterations`, where each member that has not converged lands
/// `on_limit(&cycle, current)`.
pub struct Fixpoint<T> {
    initial: Box<dyn Fn() -> T + Send + Sync>,
    converged: Option<ConvergedFn<T>>,
    step: Option<StepFn<T>>,
    max_iterations: u32,
    on_limit: Option<OnLimitFn<T>>,
}

impl<T: Value> Fixpoint<T> {
    /// A fixpoint starting from `initial()`, converging on equality,
    /// stepping to the closure's output and stopping after 100
    /// iterations at the last value.
    pub fn new(initial: impl Fn() -> T + Send + Sync + 'static) -> Self {
        Self {
            initial: Box::new(initial),
            converged: None,
            step: None,
            max_iterations: 100,
            on_limit: None,
        }
    }

    /// The convergence test between two successive values.
    pub fn converged(mut self, f: impl Fn(&T, &T) -> bool + Send + Sync + 'static) -> Self {
        self.converged = Some(Box::new(f));
        self
    }

    /// The next value from the previous one, the closure's output and
    /// the iteration number, which starts at 1.
    pub fn step(mut self, f: impl Fn(&T, T, u32) -> T + Send + Sync + 'static) -> Self {
        self.step = Some(Box::new(f));
        self
    }

    /// The iteration limit. A cycle stops at the smallest limit among
    /// its members.
    ///
    /// # Panics
    ///
    /// If `n` is 0.
    pub fn max_iterations(mut self, n: u32) -> Self {
        assert!(n > 0, "incr-core: a fixpoint needs at least one iteration");
        self.max_iterations = n;
        self
    }

    /// The value a member that has not converged lands at the limit.
    pub fn on_limit(mut self, f: impl Fn(&Cycle, T) -> T + Send + Sync + 'static) -> Self {
        self.on_limit = Some(Box::new(f));
        self
    }
}

/// Options for [`Runtime::create_query_with`](crate::Runtime::create_query_with).
pub struct QueryOpts<T> {
    pub cycle: CycleStrategy<T>,
}

impl<T> Default for QueryOpts<T> {
    fn default() -> Self {
        Self {
            cycle: CycleStrategy::None,
        }
    }
}

/// A value of the member's type, boxed so an episode can hold every
/// member's values in one place.
pub(crate) type Erased = Box<dyn Any + Send>;

/// A member's strategy with its value type erased. Every method that
/// takes an [`Erased`] is handed values of the member's own type.
pub(crate) trait ErasedStrategy: Send + Sync {
    /// The iteration limit, or `None` for a Fallback.
    fn max_iterations(&self) -> Option<u32>;
    fn v0(&self, cycle: &Cycle) -> Erased;
    fn step(&self, prev: &Erased, out: Erased, k: u32) -> Erased;
    fn converged(&self, prev: &Erased, cur: &Erased) -> bool;
    fn on_limit(&self, cycle: &Cycle, s: Erased) -> Erased;
    fn same(&self, a: &Erased, b: &Erased) -> bool;
}

pub(crate) fn downcast<T: 'static>(v: &Erased) -> &T {
    v.downcast_ref::<T>()
        .expect("incr-core: a cycle value has its member's type")
}

fn unbox<T: 'static>(v: Erased) -> T {
    *v.downcast::<T>()
        .expect("incr-core: a cycle value has its member's type")
}

struct Typed<T>(CycleStrategy<T>);

impl<T: Value> ErasedStrategy for Typed<T> {
    fn max_iterations(&self) -> Option<u32> {
        match &self.0 {
            CycleStrategy::Fixpoint(fp) => Some(fp.max_iterations),
            _ => None,
        }
    }

    fn v0(&self, cycle: &Cycle) -> Erased {
        match &self.0 {
            CycleStrategy::Fallback(g) => Box::new(g(cycle)),
            CycleStrategy::Fixpoint(fp) => Box::new((fp.initial)()),
            CycleStrategy::None => unreachable!("a strategy entry is never None"),
        }
    }

    fn step(&self, prev: &Erased, out: Erased, k: u32) -> Erased {
        match &self.0 {
            CycleStrategy::Fixpoint(Fixpoint { step: Some(f), .. }) => {
                Box::new(f(downcast::<T>(prev), unbox::<T>(out), k))
            }
            _ => out,
        }
    }

    fn converged(&self, prev: &Erased, cur: &Erased) -> bool {
        match &self.0 {
            CycleStrategy::Fixpoint(Fixpoint {
                converged: Some(f), ..
            }) => f(downcast::<T>(prev), downcast::<T>(cur)),
            _ => downcast::<T>(prev) == downcast::<T>(cur),
        }
    }

    fn on_limit(&self, cycle: &Cycle, s: Erased) -> Erased {
        match &self.0 {
            CycleStrategy::Fixpoint(Fixpoint {
                on_limit: Some(f), ..
            }) => Box::new(f(cycle, unbox::<T>(s))),
            _ => s,
        }
    }

    fn same(&self, a: &Erased, b: &Erased) -> bool {
        downcast::<T>(a) == downcast::<T>(b)
    }
}

/// The erased form of `strategy`, or `None` for [`CycleStrategy::None`].
pub(crate) fn erase<T: Value>(strategy: CycleStrategy<T>) -> Option<Arc<dyn ErasedStrategy>> {
    match strategy {
        CycleStrategy::None => None,
        s => Some(Arc::new(Typed(s))),
    }
}

/// `ComputeCtl::flags`: a recompute, whose write compares first.
pub(crate) const RECOMPUTE: u8 = 1;
/// `ComputeCtl::flags`: an episode's member run, whose value is buffered.
pub(crate) const BUFFERED: u8 = 2;
/// `ComputeCtl::flags`: the frame read a node of a cycle that is still
/// open, so its value is provisional and is buffered, not written.
pub(crate) const LOW: u8 = 4;
/// `ComputeCtl::flags`: a dep read in the frame did not land `Clean`.
pub(crate) const POISONED: u8 = 8;

/// What one compute frame tells its wrapper and its landing. It lives on
/// the stack of the code that runs the closure, and the frame points at
/// it while the closure runs.
#[doc(hidden)]
pub struct ComputeCtl {
    pub(crate) flags: Cell<u8>,
    /// The lowest conceptual-stack position this frame reached through a
    /// cycle read, as in Tarjan's algorithm. Starts at the frame's own.
    pub(crate) lowlink: Cell<u32>,
    pub(crate) buf: Cell<Option<Erased>>,
    /// Set once the slot holds the new value.
    pub(crate) wrote: Cell<u8>,
}

impl ComputeCtl {
    #[inline(always)]
    pub(crate) fn new(flags: u8) -> Self {
        Self {
            flags: Cell::new(flags),
            lowlink: Cell::new(u32::MAX),
            buf: Cell::new(None),
            wrote: Cell::new(0),
        }
    }

    #[inline(always)]
    pub(crate) fn has(&self, flag: u8) -> bool {
        self.flags.get() & flag != 0
    }

    #[inline(always)]
    pub(crate) fn raise(&self, flag: u8) {
        self.flags.set(self.flags.get() | flag);
    }

    /// Marks the frame part of an open cycle reaching down to `p`.
    pub(crate) fn lower(&self, p: u32) {
        self.raise(LOW);
        self.lowlink.set(self.lowlink.get().min(p));
    }

    /// The wrapper's cold arm: keep the value for the episode.
    #[cold]
    #[inline(never)]
    pub(crate) fn buffer<T: Value>(&self, value: T) -> Outcome {
        self.buf.set(Some(Box::new(value)));
        Outcome::Buffered
    }
}

/// What a compute closure's wrapper did with the value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Written, and different from the slot's previous value.
    Changed,
    /// Equal to the slot's value, so nothing was written.
    Unchanged,
    /// Kept in `ComputeCtl::buf`, not written.
    Buffered,
}

/// The payload of an engine-raised cycle unwind. Its kind is recorded in
/// the thread's unwind kind, never read from the payload type.
pub(crate) struct CycleUnwind(pub(crate) Cycle);

/// The payload of a wound: the wound-wait scheduler releasing this
/// thread's claims so an older thread can finish.
pub(crate) struct WoundUnwind;
