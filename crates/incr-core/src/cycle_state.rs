//! Per-thread cycle state shared by every runtime on the thread: the
//! claims this thread holds outside its frame stack (the held map), the
//! engine entry depth, the kind of an engine unwind in flight, the last
//! cycle an uncaught cycle panic reported, and the roots whose episodes
//! dissolved during the current outermost entry.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use std::sync::Arc;

use crate::cycle::{Cycle, Erased, ErasedStrategy};
use crate::handle::RuntimeId;
use crate::node::NodeId;
use crate::value_code;

/// A claim this thread holds outside its frame stack: a node deferred
/// during discovery, or a member of a running episode.
pub(crate) struct Held {
    pub(crate) generation: u32,
    /// The conceptual-stack position of the root that resolves it.
    pub(crate) anchor: u32,
    /// The position of the frame whose unwind releases it: the frame it
    /// stands for, or the frame below that its deferral passed to. A
    /// deferral to a root below that frame leaves it anchored lower, so the
    /// anchor alone would let an unwind between them leak the claim.
    pub(crate) owner: u32,
    /// The node of the frame at `owner`, `u32::MAX` for any: another frame
    /// at that position later, a sibling's, owns none of it.
    pub(crate) owner_node: u32,
    /// The owner is a virtual frame, the run of `owner_node` that a doom
    /// discarded, with that frame's path. Another run of the node at the
    /// position, as its reader's retry makes when it reads the node more
    /// than once, has another path and must not take or release what the
    /// doomed run deferred.
    pub(crate) owner_virtual: Option<u64>,
    /// For a virtual owner, the position and path of each virtual frame
    /// below it that has popped since: a run at such a position along
    /// another path is not the owner's ancestor.
    pub(crate) ancestry: Vec<(u32, u64)>,
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
    /// The last run reached the node's child index.
    pub(crate) opened: bool,
    /// The landing changed the slot's value.
    pub(crate) changed: bool,
    pub(crate) poisoned: bool,
    /// A run of this claim panicked in user code.
    pub(crate) failed: bool,
    /// `Cycle(M)` while the node is a member of a running episode.
    pub(crate) episode: Option<Cycle>,
    /// That episode's id, 0 when there is none.
    pub(crate) ep: u64,
    /// The member is in the episode's set of unseeded members that read
    /// each other.
    pub(crate) in_cu: bool,
}

impl Held {
    /// Hands the claim to the frame at `owner`, of `node` (`u32::MAX` for
    /// whichever frame is there), virtual at a path or not.
    pub(crate) fn own(&mut self, owner: u32, node: u32, virtual_frame: Option<u64>) {
        self.owner = owner;
        self.owner_node = node;
        self.owner_virtual = virtual_frame;
        self.ancestry.clear();
    }

    /// The run at `cidx`, of `node` (`u32::MAX` for any), at `path` (the
    /// path when a virtual frame at or above `cidx` owns claims), owns the
    /// claim: one owned at its position by it, or owned above it in its
    /// subtree.
    pub(crate) fn owned_by(&self, cidx: u32, node: u32, path: Option<u64>) -> bool {
        let at = |p: u64| Some(p) == path;
        let of_node = node == u32::MAX || self.owner_node == node || self.owner_node == u32::MAX;
        match self.owner_virtual {
            _ if self.owner < cidx => false,
            None => self.owner > cidx || of_node,
            Some(p) if self.owner == cidx => at(p) && of_node,
            Some(_) => self
                .ancestry
                .iter()
                .find(|a| a.0 == cidx)
                .is_none_or(|a| at(a.1)),
        }
    }

    pub(crate) fn new(generation: u32, anchor: u32, seeded: bool, first: bool) -> Self {
        Self {
            generation,
            anchor,
            owner: anchor,
            owner_node: u32::MAX,
            owner_virtual: None,
            ancestry: Vec::new(),
            seeded,
            first,
            buf: None,
            s: None,
            deps: Vec::new(),
            last: Vec::new(),
            installed: Vec::new(),
            wrote: false,
            opened: false,
            changed: false,
            poisoned: false,
            failed: false,
            episode: None,
            ep: 0,
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

/// An episode between member runs. A doom that discards a member run
/// suspends it in a replay context until that run's virtual frame pops.
pub(crate) struct Episode {
    pub(crate) id: u64,
    pub(crate) root: NodeId,
    pub(crate) cidx: u32,
    pub(crate) validated: bool,
    pub(crate) members: Vec<u32>,
    pub(crate) cu: Vec<u32>,
    /// The iteration state since the last restart, `None` at a restart.
    pub(crate) round: Option<Round>,
}

pub(crate) struct Round {
    pub(crate) cycle: Cycle,
    pub(crate) strategies: Vec<(u32, Arc<dyn ErasedStrategy>)>,
    pub(crate) limit: u32,
    pub(crate) unseeded: Vec<u32>,
    pub(crate) stamp_rev: u64,
    pub(crate) verify_rev: u64,
    /// `s_{k-1}` of each Fixpoint member.
    pub(crate) prev: Vec<Option<Erased>>,
    pub(crate) k: u32,
    pub(crate) low: u32,
    pub(crate) step: Step,
}

/// Where iteration `k` of a round stands.
pub(crate) enum Step {
    Clear,
    /// The seeded member at this index runs next.
    Seeded(usize),
    /// The unseeded member at this index runs next, if no read has.
    Unseeded(usize),
    /// Steps 6, 5, validation, CU growth and 4.
    Check,
    /// The landing pass's runs of the unseeded members outside CU.
    Landing {
        outside: Vec<u32>,
        next: usize,
    },
}

/// Why an engine unwind is in flight.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum UnwindKind {
    /// A read under `get` closed a cycle that has no value for it.
    Cycle,
    /// The wound-wait scheduler released this thread's claims.
    Wound,
    /// A user panic in a speculative entry's run. Every frame it reaches,
    /// a closure that caught it included, takes the discard landing, so
    /// no value computed over the swallowed panic lands.
    Discard,
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
    /// The held map has entries: a check of a thread that holds nothing
    /// reads this block and not the map, whose access tests liveness.
    held_any: Cell<bool>,
    /// The address of a local of the outermost entry, from which a read
    /// measures the stack it has used against a byte budget.
    stack_base: Cell<usize>,
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
            held_any: Cell::new(false),
            stack_base: Cell::new(0),
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
        held_any: Cell::new(false),
        stack_base: Cell::new(0),
    };
    static OUTER: RefCell<Outer> = RefCell::new(Outer {
        dissolved: Vec::new(),
        last_cycle: None,
    });
}

/// Runs `f` on the held map. `f` must not call user code or the engine.
pub(crate) fn with_held<R>(f: impl FnOnce(&mut HeldMap) -> R) -> R {
    HELD.with(|h| {
        let mut h = h.borrow_mut();
        // Set first, so that a panic inside `f` leaves it set: a stale
        // true costs a lookup, a stale false would hide a claim.
        ENTRY.with(|e| e.held_any.set(true));
        let r = f(&mut h);
        ENTRY.with(|e| e.held_any.set(!h.is_empty()));
        r
    })
}

fn held_any() -> bool {
    ENTRY.with(|e| e.held_any.get())
}

/// Every claim in the held map, of every runtime.
pub(crate) fn held_keys() -> Vec<(RuntimeId, u32)> {
    HELD.with(|h| h.borrow().keys().copied().collect())
}

pub(crate) fn holds(rt: RuntimeId, slot: u32) -> bool {
    held_any() && HELD.with(|h| h.borrow().contains_key(&(rt, slot)))
}

/// The anchor of `slot`'s held claim, if this thread holds one.
pub(crate) fn held_anchor(rt: RuntimeId, slot: u32) -> Option<u32> {
    if !held_any() {
        return None;
    }
    HELD.with(|h| h.borrow().get(&(rt, slot)).map(|e| e.anchor))
}

impl UnwindKind {
    const fn bits(kind: Option<UnwindKind>) -> u64 {
        (match kind {
            None => 0,
            Some(UnwindKind::Cycle) => 1,
            Some(UnwindKind::Wound) => 2,
            Some(UnwindKind::Discard) => 3,
        }) << value_code::UNWIND_SHIFT
    }

    /// The kind held in a thread word.
    pub(crate) fn of_word(word: u64) -> Option<UnwindKind> {
        match (word & value_code::UNWIND_MASK) >> value_code::UNWIND_SHIFT {
            0 => None,
            1 => Some(UnwindKind::Cycle),
            2 => Some(UnwindKind::Wound),
            _ => Some(UnwindKind::Discard),
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
/// `get` raised, `Some(None)` for a wound or a speculative discard,
/// `None` for anything else.
#[doc(hidden)]
pub fn engine_unwind(payload: &(dyn std::any::Any + Send)) -> Option<Option<Cycle>> {
    match unwind_kind()? {
        UnwindKind::Cycle => payload
            .downcast_ref::<crate::cycle::CycleUnwind>()
            .map(|u| Some(u.0.clone())),
        UnwindKind::Wound | UnwindKind::Discard => Some(None),
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
    count_failure();
}

/// Counts a failure already marked, as a panic of a marked member that
/// reaches its reader again.
pub(crate) fn count_failure() {
    ENTRY.with(|e| e.failures.set(e.failures.get() + 1));
}

#[cfg(all(test, not(loom)))]
thread_local! {
    static OUTERMOST_ENTRIES: Cell<u64> = const { Cell::new(0) };
}

/// The thread's 0 to 1 transitions of the entry depth, each of which
/// stored the stack base.
#[cfg(all(test, not(loom)))]
pub(crate) fn outermost_entries() -> u64 {
    OUTERMOST_ENTRIES.with(Cell::get)
}

/// The stack address the outermost engine entry recorded.
pub(crate) fn stack_base() -> usize {
    ENTRY.with(|e| e.stack_base.get())
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
        let base = 0_u8;
        let base = std::ptr::from_ref(&base) as usize;
        let depth = ENTRY.with(|e| {
            let depth = e.depth.get();
            e.depth.set(depth + 1);
            if depth == 0 {
                e.stack_base.set(base);
                #[cfg(all(test, not(loom)))]
                OUTERMOST_ENTRIES.with(|n| n.set(n.get() + 1));
            }
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
