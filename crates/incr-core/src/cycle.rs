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
use std::mem::MaybeUninit;
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
    pub replay: Replay,
}

impl<T> Default for QueryOpts<T> {
    fn default() -> Self {
        Self {
            cycle: CycleStrategy::None,
            replay: Replay::Native,
        }
    }
}

/// Whether a query's compute may be cut short and replayed when a read
/// in it would nest deeper than the runtime's frame or byte budget
/// ([`Runtime::set_max_frames`](crate::Runtime::set_max_frames)).
///
/// A replayable query's read past the budget returns its type's
/// placeholder ([`Runtime::set_placeholder`](crate::Runtime::set_placeholder))
/// instead of nesting; the run is then discarded, and a driver computes
/// the node that read wanted at a shallow depth before running the query
/// again. The results are those of the unbounded run. A query's closure
/// may therefore run more than once for one read, under the contract in
/// [`Runtime`](crate::Runtime)'s docs.
///
/// A run's reads of another runtime's nodes are kept, each value by a
/// clone, so the rerun takes what the cut-short run read rather than
/// reading the other runtime again. A read that ended in a panic whose
/// payload can't be copied keeps the run from being cut short after it
/// instead; its reads nest natively, so a deep chain that alternates
/// between two runtimes through such reads recurses past the budget.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Replay {
    /// Reads nest natively at any depth.
    #[default]
    Native,
    /// Reads past the budget unwind and replay.
    Allowed,
}

/// A value of the member's type, boxed so an episode can hold every
/// member's values in one place.
pub(crate) type Erased = Box<dyn Any + Send>;

/// A member's strategy with its value type erased. Every method that
/// takes an [`Erased`] is handed values of the member's own type.
pub(crate) trait ErasedStrategy: Send + Sync {
    /// The iteration limit, or `None` for a Fallback.
    fn max_iterations(&self) -> Option<u32>;
    /// The initial value: a Fallback's `g` of `cycle`, which only a
    /// Fallback is handed, or a Fixpoint's initial value.
    fn v0(&self, cycle: Option<&Cycle>) -> Erased;
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

    fn v0(&self, cycle: Option<&Cycle>) -> Erased {
        match &self.0 {
            CycleStrategy::Fallback(g) => {
                Box::new(g(cycle.expect("incr-core: a Fallback is handed its cycle")))
            }
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

/// A query's options with its value type erased: the strategy, and the
/// node flags the options set.
pub(crate) struct ErasedOpts {
    pub(crate) strategy: Option<Arc<dyn ErasedStrategy>>,
    pub(crate) flags: u8,
}

impl ErasedOpts {
    pub(crate) const NONE: Self = Self {
        strategy: None,
        flags: 0,
    };
}

pub(crate) fn erase_opts<T: Value>(opts: QueryOpts<T>) -> ErasedOpts {
    ErasedOpts {
        strategy: erase(opts.cycle),
        flags: match opts.replay {
            Replay::Native => 0,
            Replay::Allowed => crate::segmented_nodes::REPLAYABLE,
        },
    }
}

/// `ComputeCtl::flags`: a recompute, whose write compares first.
pub(crate) const RECOMPUTE: u16 = 1;
/// `ComputeCtl::flags`: an episode's member run, whose value is buffered.
pub(crate) const BUFFERED: u16 = 2;
/// `ComputeCtl::flags`: the frame read a node of a cycle that is still
/// open, so its value is provisional and is buffered, not written.
pub(crate) const LOW: u16 = 4;
/// `ComputeCtl::flags`: a dep read in the frame did not land `Clean`.
pub(crate) const POISONED: u16 = 8;
/// `ComputeCtl::flags`: the frame is doomed, so a table miss in it
/// returns a pending handle and creates nothing.
pub(crate) const DOOMED: u16 = 16;
/// `ComputeCtl::flags`: a read in the frame returned a placeholder, so
/// every node it reads after is reached through a suspect edge.
pub(crate) const PH_SEEN: u16 = 32;
/// `ComputeCtl::flags`: a read in the frame returned a value a landing
/// had left Dirty, or took an unwind, which `replay::keep_read` keeps for
/// a retry.
pub(crate) const KEPT_READ: u16 = 64;
/// A member run at its anchor, below the frames that read it, or a frame
/// within the budget that took a payload no kept read carries: it is never
/// doomed, and marking stops at it.
pub(crate) const NO_DOOM: u16 = 128;
/// `ComputeCtl::flags`: the compute is a walk's entry, or a replay of
/// one, which contains a panic out of it (`compute_panicked`).
pub(crate) const WALKED: u16 = 256;
/// `ComputeCtl::flags`: the frame read another runtime's node with an
/// outcome no kept read carries. Its own runtime treats it as `NO_DOOM`
/// within that runtime's budget only, which the other runtime can't see.
pub(crate) const FOREIGN_HOLD: u16 = 512;
/// `ComputeCtl::flags`: a speculative run, which the unbounded run may never
/// make: its value is buffered and its landing withheld (`replay::Stash`).
pub(crate) const SPECULATIVE: u16 = 1024;
/// `ComputeCtl::flags`: a read in the frame gave an outcome that holds
/// until the next set (the panic of a node that landed Failed, or a
/// memo's value). Without `POISONED` the frame lands Dirty with a memo
/// for the revision (`NodeData::set_memo`).
pub(crate) const HANDLED: u16 = 2048;
/// `ComputeCtl::flags`: the run read a value no rerun of its node reads
/// again, as a Dirty node's slot, a stash, a cycle, or another runtime's
/// node, so its panic is not regenerated (`Runtime::regenerate`).
pub(crate) const UNREGENERABLE: u16 = 4096;
/// `ComputeCtl::flags`: the frame made a read its stash adoption can't
/// check, so a speculative run of it is not stashed:
/// - a kept read (`KEPT_READ`) of its own runtime, or of another runtime's
///   node that was not Clean or was read while that runtime was changing;
/// - on a `SPECULATIVE` frame, a read of another runtime it neither keeps
///   nor records, or holds instead of keeping (`FOREIGN_HOLD`), and a
///   lookup that records nothing;
/// - a read of a deleted node's handle.
pub(crate) const KEPT_UNCHECKED: u16 = 8192;

/// `ComputeCtl::flags`: the frame read a landing this thread's attempt
/// holds (`crate::hold`), so its own landing is held too, and so is its
/// reader's.
pub(crate) const HOLD: u16 = 16384;
/// A `HOLD` frame's write kept what it displaced in `crate::hold`, keyed
/// by this control block, until the landing's entry takes it or the frame
/// ends (`crate::hold::KeptGuard`).
pub(crate) const KEPT: u16 = 32768;
pub(crate) const RECOMPUTE_HOLD: u16 = RECOMPUTE | HOLD;
/// A speculative frame whose outcome no stash keeps: a doomed run is
/// discarded, and an open cycle's provisional read may settle otherwise.
pub(crate) const UNSTASHED: u16 = DOOMED | LOW;

const _: () = assert!(crate::bits::disjoint(&[
    RECOMPUTE as u64,
    BUFFERED as u64,
    LOW as u64,
    POISONED as u64,
    DOOMED as u64,
    PH_SEEN as u64,
    KEPT_READ as u64,
    NO_DOOM as u64,
    WALKED as u64,
    FOREIGN_HOLD as u64,
    SPECULATIVE as u64,
    HANDLED as u64,
    UNREGENERABLE as u64,
    KEPT_UNCHECKED as u64,
    HOLD as u64,
    KEPT as u64,
]));

/// What one compute frame tells its wrapper and its landing. It lives on
/// the stack of the code that runs the closure, and the frame points at
/// it while the closure runs.
#[doc(hidden)]
pub struct ComputeCtl {
    pub(crate) flags: Cell<u16>,
    /// The lowest conceptual-stack position this frame reached through a
    /// cycle read, as in Tarjan's algorithm. Starts at the frame's own.
    pub(crate) lowlink: Cell<u32>,
    pub(crate) buf: Cell<Option<Erased>>,
    /// Set once the slot holds the new value.
    pub(crate) wrote: Cell<u8>,
    /// Whether `path` holds the frame's path.
    path_kept: Cell<bool>,
    /// The creator's `ChildIndex` while this run may reach it, or null
    /// until the run creates its first child.
    pub(crate) children: Cell<*const ()>,
    /// The frame's runtime's [`KeepResolver`], or null for a frame that
    /// settles its keeping when it is pushed or keeps nothing.
    keep: *const KeepResolver,
    /// The frame's path once found (`path_kept`), left unwritten until
    /// then so a push pays nothing for it.
    path: Cell<MaybeUninit<u64>>,
}

/// Lives inside its runtime and answers whether a frame of node `slot` of
/// that runtime keeps other runtimes' Clean reads for its retry. The
/// runtime's `keep_resolver_ptr` is good while one of its computes runs,
/// since the compute borrows the runtime; a pointer taken from the field
/// itself may not reach the rest of the runtime.
pub(crate) struct KeepResolver {
    pub(crate) keeps: unsafe fn(*const KeepResolver, u32) -> bool,
}

impl ComputeCtl {
    #[inline(always)]
    pub(crate) fn new(flags: u16) -> Self {
        Self::resolved_by(flags, std::ptr::null())
    }

    #[inline(always)]
    pub(crate) fn resolved_by(flags: u16, keep: *const KeepResolver) -> Self {
        Self {
            flags: Cell::new(flags),
            lowlink: Cell::new(u32::MAX),
            buf: Cell::new(None),
            wrote: Cell::new(0),
            path_kept: Cell::new(false),
            children: Cell::new(std::ptr::null()),
            keep,
            path: Cell::new(MaybeUninit::uninit()),
        }
    }

    /// Whether this frame, of node `slot`, keeps other runtimes' Clean
    /// reads.
    ///
    /// # Safety
    ///
    /// The frame must be running, so that its runtime is borrowed.
    pub(crate) unsafe fn keeps_foreign(&self, slot: u32) -> bool {
        // SAFETY: the caller's frame is running, so the runtime holding
        // the resolver is borrowed and in place.
        !self.keep.is_null() && unsafe { ((*self.keep).keeps)(self.keep, slot) }
    }

    /// The frame's path, if [`Self::keep_path`] kept it.
    pub(crate) fn kept_path(&self) -> Option<u64> {
        // SAFETY: `keep_path` writes `path` before it sets `path_kept`.
        self.path_kept
            .get()
            .then(|| unsafe { self.path.get().assume_init() })
    }

    /// Keeps `path` as the frame's path, which the frames below it fix
    /// while it is on the stack.
    pub(crate) fn keep_path(&self, path: u64) {
        self.path.set(MaybeUninit::new(path));
        self.path_kept.set(true);
    }

    #[inline(always)]
    pub(crate) fn has(&self, flag: u16) -> bool {
        self.flags.get() & flag != 0
    }

    /// The frame's `POISONED` and `HANDLED` flags.
    pub(crate) fn poison(&self) -> u16 {
        self.flags.get() & (POISONED | HANDLED)
    }

    #[inline(always)]
    pub(crate) fn raise(&self, flag: u16) {
        self.flags.set(self.flags.get() | flag);
    }

    pub(crate) fn clear(&self, flag: u16) {
        self.flags.set(self.flags.get() & !flag);
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

/// The payload a speculative entry's discard unwind carries on from a
/// closure that swallowed it.
pub(crate) struct DiscardUnwind;
