//! Per-thread cycle state shared by every runtime on the thread: the
//! claims this thread holds outside its frame stack (the held map), the
//! engine entry depth, the kind of an engine unwind in flight, the last
//! cycle an uncaught cycle panic reported, and the roots whose episodes
//! dissolved during the current outermost entry.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

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

/// The engine entry state of a thread, in one block so that an entry
/// and its exit touch one thread-local.
struct EntryState {
    depth: Cell<u32>,
    /// This thread's wound-wait ticket, 0 until its first registration
    /// inside the current outermost entry, which keeps it across retries.
    ticket: Cell<u64>,
    /// Set by a wound until the outermost entry retries.
    wounded: Cell<bool>,
    failures: Cell<u64>,
    /// Set when `Outer` may hold something, so that an outermost entry
    /// with nothing to reset touches only this block, which has no
    /// destructor and so no liveness check.
    outer_set: Cell<bool>,
}

/// What the outermost entry resets: the roots whose episodes dissolved,
/// and the cycle the last uncaught cycle panic reported.
struct Outer {
    dissolved: Vec<(RuntimeId, u32)>,
    last_cycle: Option<Cycle>,
}

#[cfg(not(loom))]
thread_local! {
    static HELD: RefCell<HeldMap> = RefCell::new(HashMap::new());
    static ENTRY: EntryState = const {
        EntryState {
            depth: Cell::new(0),
            ticket: Cell::new(0),
            wounded: Cell::new(false),
            failures: Cell::new(0),
            outer_set: Cell::new(false),
        }
    };
    static OUTER: RefCell<Outer> = const {
        RefCell::new(Outer {
            dissolved: Vec::new(),
            last_cycle: None,
        })
    };
}

#[cfg(loom)]
loom::thread_local! {
    static HELD: RefCell<HeldMap> = RefCell::new(HashMap::new());
    static ENTRY: EntryState = EntryState {
        depth: Cell::new(0),
        ticket: Cell::new(0),
        wounded: Cell::new(false),
        failures: Cell::new(0),
        outer_set: Cell::new(false),
    };
    static OUTER: RefCell<Outer> = RefCell::new(Outer {
        dissolved: Vec::new(),
        last_cycle: None,
    });
}

/// Runs `f` on the held map. `f` must not call user code or the engine.
pub(crate) fn with_held<R>(f: impl FnOnce(&mut HeldMap) -> R) -> R {
    HELD.with(|h| f(&mut h.borrow_mut()))
}

/// Every claim in the held map, of every runtime.
pub(crate) fn held_keys() -> Vec<(RuntimeId, u32)> {
    HELD.with(|h| h.borrow().keys().copied().collect())
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

/// For a binding whose boundary catches every unwind, such as one that
/// runs a closure in another language: whether `payload` is an engine
/// unwind in flight, which the boundary must carry through the other
/// language and resume unchanged. `Some(Some(cycle))` for a cycle a
/// `get` raised, `Some(None)` for a wound, `None` for anything else.
#[doc(hidden)]
pub fn engine_unwind(payload: &(dyn std::any::Any + Send)) -> Option<Option<Cycle>> {
    match unwind_kind()? {
        UnwindKind::Cycle => payload
            .downcast_ref::<crate::cycle::CycleUnwind>()
            .map(|u| Some(u.0.clone())),
        UnwindKind::Wound => Some(None),
    }
}

/// Clears the last reported cycle and returns it.
pub fn take_last_cycle() -> Option<Cycle> {
    OUTER.with(|o| o.borrow_mut().last_cycle.take())
}

pub(crate) fn set_last_cycle(cycle: Cycle) {
    ENTRY.with(|e| e.outer_set.set(true));
    OUTER.with(|o| o.borrow_mut().last_cycle = Some(cycle));
}

/// Whether `root` already dissolved an episode in this outermost entry,
/// marking it if not.
pub(crate) fn dissolved_before(rt: RuntimeId, root: u32) -> bool {
    OUTER.with(|o| {
        let dissolved = &mut o.borrow_mut().dissolved;
        if dissolved.contains(&(rt, root)) {
            return true;
        }
        dissolved.push((rt, root));
        ENTRY.with(|e| e.outer_set.set(true));
        false
    })
}

pub(crate) fn has_dissolved(rt: RuntimeId, root: u32) -> bool {
    OUTER.with(|o| o.borrow().dissolved.contains(&(rt, root)))
}

/// How many held claims this thread has marked failed: a member run that
/// panics marks its node only if no member nested in it was marked, so
/// the panic is attributed to the member whose own code raised it.
pub(crate) fn failures() -> u64 {
    ENTRY.with(|e| e.failures.get())
}

/// Marks `held` failed.
pub(crate) fn mark_failed(held: &mut Held) {
    held.failed = true;
    ENTRY.with(|e| e.failures.set(e.failures.get() + 1));
}

/// The thread's engine entry depth: 0 outside every engine entry.
pub(crate) fn depth() -> u32 {
    ENTRY.with(|e| e.depth.get())
}

/// This thread's wound-wait ticket, taken from `next` at its first
/// registration inside the outermost entry.
pub(crate) fn ticket(next: impl FnOnce() -> u64) -> u64 {
    ENTRY.with(|e| {
        if e.ticket.get() == 0 {
            e.ticket.set(next());
        }
        e.ticket.get()
    })
}

pub(crate) fn set_wounded(wounded: bool) {
    ENTRY.with(|e| e.wounded.set(wounded));
}

pub(crate) fn wounded() -> bool {
    ENTRY.with(|e| e.wounded.get())
}

/// One count on the thread's engine entry depth; dropping it, by an
/// unwind included, gives the count back, and the outermost entry's drop
/// gives up the thread's ticket.
pub(crate) struct EntryDepth {
    outermost: bool,
}

impl EntryDepth {
    #[inline(always)]
    pub(crate) fn enter() -> Self {
        let depth = ENTRY.with(|e| {
            let depth = e.depth.get();
            e.depth.set(depth + 1);
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
        ENTRY.with(|e| {
            let depth = e.depth.get() - 1;
            e.depth.set(depth);
            if depth == 0 {
                e.ticket.set(0);
            }
        });
    }
}

/// Resets the state an outermost entry starts from.
#[inline(always)]
pub(crate) fn reset_outermost() {
    if ENTRY.with(|e| e.outer_set.replace(false)) {
        OUTER.with(|o| {
            let mut o = o.borrow_mut();
            o.dissolved.clear();
            o.last_cycle = None;
        });
    }
}
