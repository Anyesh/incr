//! Per-thread cycle state shared by every runtime on the thread: the
//! claims this thread holds outside its frame stack (the held map), the
//! engine entry depth, the kind of an engine unwind in flight, the last
//! cycle an uncaught cycle panic reported, and the roots whose episodes
//! dissolved during the current outermost entry.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use crate::cycle::{Cycle, Erased};
use crate::handle::RuntimeId;
use crate::node::NodeId;
use crate::value_code;

/// A claim this thread holds outside its frame stack: a node deferred
/// during discovery, or a member of a running episode.
pub(crate) struct Held {
    pub(crate) generation: u32,
    /// The conceptual-stack position of the root that resolves it.
    pub(crate) anchor: u32,
    /// The node has a Fallback or Fixpoint strategy.
    pub(crate) seeded: bool,
    /// The claim was taken from New.
    pub(crate) first: bool,
    /// The last run's value, or for a seeded member its last output.
    pub(crate) buf: Option<Erased>,
    /// A seeded member's current value inside an episode.
    pub(crate) s: Option<Erased>,
    /// Every dep any run of this claim recorded, for the stale check.
    pub(crate) deps: Vec<(NodeId, u32)>,
    /// The deps of the last run: its reads of other members are the
    /// iteration's graph.
    pub(crate) last: Vec<(NodeId, u32)>,
    /// Every dep the runs since the episode's last restart recorded, in
    /// first-read order: the landed value depends on all of them, so the
    /// landing installs them all.
    pub(crate) installed: Vec<NodeId>,
    pub(crate) wrote: bool,
    /// The landing changed the slot's value.
    pub(crate) changed: bool,
    pub(crate) poisoned: bool,
    /// A run of this claim panicked in user code.
    pub(crate) failed: bool,
    /// `Cycle(M)` while the node is a member of a running episode.
    pub(crate) episode: Option<Cycle>,
    /// The member is in the episode's set of unseeded members that read
    /// each other.
    pub(crate) in_cu: bool,
}

impl Held {
    pub(crate) fn new(generation: u32, anchor: u32, seeded: bool, first: bool) -> Self {
        Self {
            generation,
            anchor,
            seeded,
            first,
            buf: None,
            s: None,
            deps: Vec::new(),
            last: Vec::new(),
            installed: Vec::new(),
            wrote: false,
            changed: false,
            poisoned: false,
            failed: false,
            episode: None,
            in_cu: false,
        }
    }

    /// Records one run's deps.
    pub(crate) fn record_run(&mut self, deps: Vec<(NodeId, u32)>) {
        for &d in &deps {
            if !self.deps.contains(&d) {
                self.deps.push(d);
            }
            if !self.installed.contains(&d.0) {
                self.installed.push(d.0);
            }
        }
        self.last = deps;
    }
}

/// Why an engine unwind is in flight.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum UnwindKind {
    /// A read under `get` closed a cycle that has no value for it.
    Cycle,
    /// The wound-wait scheduler released this thread's claims.
    Wound,
}

type HeldMap = HashMap<(RuntimeId, u32), Held>;

#[cfg(not(loom))]
thread_local! {
    static HELD: RefCell<HeldMap> = RefCell::new(HashMap::new());
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    static FAILURES: Cell<u64> = const { Cell::new(0) };
    static LAST_CYCLE: RefCell<Option<Cycle>> = const { RefCell::new(None) };
    static DISSOLVED: RefCell<HashSet<(RuntimeId, u32)>> = RefCell::new(HashSet::new());
}

#[cfg(loom)]
loom::thread_local! {
    static HELD: RefCell<HeldMap> = RefCell::new(HashMap::new());
    static DEPTH: Cell<u32> = Cell::new(0);
    static FAILURES: Cell<u64> = Cell::new(0);
    static LAST_CYCLE: RefCell<Option<Cycle>> = RefCell::new(None);
    static DISSOLVED: RefCell<HashSet<(RuntimeId, u32)>> = RefCell::new(HashSet::new());
}

/// Runs `f` on the held map. `f` must not call user code or the engine.
pub(crate) fn with_held<R>(f: impl FnOnce(&mut HeldMap) -> R) -> R {
    HELD.with(|h| f(&mut h.borrow_mut()))
}

pub(crate) fn holds(rt: RuntimeId, slot: u32) -> bool {
    HELD.with(|h| {
        let h = h.borrow();
        !h.is_empty() && h.contains_key(&(rt, slot))
    })
}

/// The anchor of `slot`'s held claim, if this thread holds one.
pub(crate) fn held_anchor(rt: RuntimeId, slot: u32) -> Option<u32> {
    HELD.with(|h| h.borrow().get(&(rt, slot)).map(|e| e.anchor))
}

impl UnwindKind {
    const fn bits(kind: Option<UnwindKind>) -> u64 {
        (match kind {
            None => 0,
            Some(UnwindKind::Cycle) => 1,
            Some(UnwindKind::Wound) => 2,
        }) << value_code::UNWIND_SHIFT
    }

    /// The kind held in a thread word.
    pub(crate) fn of_word(word: u64) -> Option<UnwindKind> {
        match (word & value_code::UNWIND_MASK) >> value_code::UNWIND_SHIFT {
            0 => None,
            1 => Some(UnwindKind::Cycle),
            _ => Some(UnwindKind::Wound),
        }
    }
}

pub(crate) fn unwind_kind() -> Option<UnwindKind> {
    UnwindKind::of_word(value_code::thread_word())
}

pub(crate) fn set_unwind_kind(kind: Option<UnwindKind>) {
    value_code::set_unwind_bits(UnwindKind::bits(kind));
}

pub(crate) fn take_unwind_kind() -> Option<UnwindKind> {
    let kind = unwind_kind();
    if kind.is_some() {
        set_unwind_kind(None);
    }
    kind
}

/// Clears the last reported cycle and returns it.
pub fn take_last_cycle() -> Option<Cycle> {
    LAST_CYCLE.with(|c| c.borrow_mut().take())
}

pub(crate) fn set_last_cycle(cycle: Cycle) {
    LAST_CYCLE.with(|c| *c.borrow_mut() = Some(cycle));
}

/// Whether `root` already dissolved an episode in this outermost entry,
/// marking it if not.
pub(crate) fn dissolved_before(rt: RuntimeId, root: u32) -> bool {
    DISSOLVED.with(|d| !d.borrow_mut().insert((rt, root)))
}

pub(crate) fn has_dissolved(rt: RuntimeId, root: u32) -> bool {
    DISSOLVED.with(|d| d.borrow().contains(&(rt, root)))
}

/// How many held claims this thread has marked failed: a member run that
/// panics marks its node only if no member nested in it was marked, so
/// the panic is attributed to the member whose own code raised it.
pub(crate) fn failures() -> u64 {
    FAILURES.with(Cell::get)
}

/// Marks `held` failed.
pub(crate) fn mark_failed(held: &mut Held) {
    held.failed = true;
    FAILURES.with(|f| f.set(f.get() + 1));
}

/// The thread's engine entry depth: 0 outside every engine entry.
pub(crate) fn depth() -> u32 {
    DEPTH.with(Cell::get)
}

/// One count on the thread's engine entry depth; dropping it, by an
/// unwind included, gives the count back.
pub(crate) struct EntryDepth {
    outermost: bool,
}

impl EntryDepth {
    #[inline(always)]
    pub(crate) fn enter() -> Self {
        let depth = DEPTH.with(|d| {
            let depth = d.get();
            d.set(depth + 1);
            depth
        });
        Self {
            outermost: depth == 0,
        }
    }

    #[inline(always)]
    pub(crate) fn is_outermost(&self) -> bool {
        self.outermost
    }
}

impl Drop for EntryDepth {
    #[inline(always)]
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Resets the state an outermost entry starts from.
#[cold]
pub(crate) fn reset_outermost() {
    DISSOLVED.with(|d| d.borrow_mut().clear());
    LAST_CYCLE.with(|c| c.borrow_mut().take());
    set_unwind_kind(None);
}
