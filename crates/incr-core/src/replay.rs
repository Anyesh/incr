//! Per-thread replay state: the contexts a doom opens and their drivers'
//! work. A read past a runtime's frame budget in a replayable frame
//! dooms the frames above the context's boundary that can take a
//! placeholder: each becomes a virtual frame on the context's `vstack`,
//! standing for the frame the unbounded run has there, and the node the
//! read wanted becomes an entry one level above the reading frame's. The
//! context's driver runs entries highest level first, each as a fresh
//! root at its level's position, until its level-0 target lands.
//!
//! Every context of every runtime on the thread lives in one stack, in
//! the order they nest; a runtime's contexts are those with its id.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};

use crate::cycle_state::Episode;
use crate::handle::RuntimeId;
use crate::taint_memo;
use crate::value_code;

/// A node the driver is to run at a level.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Entry {
    pub(crate) node: u32,
    pub(crate) generation: u32,
    /// The edge from the frame that pushed it was suspect: that frame had
    /// read a placeholder before it read this node.
    pub(crate) suspect: bool,
    /// The entry lies above a suspect edge, in this context or an
    /// enclosing one, so the unbounded run may never evaluate it.
    pub(crate) spec: bool,
    pub(crate) retry: Retry,
}

/// How the driver runs an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Retry {
    /// A read: its walk, rooted at the entry.
    Walk,
    /// A compute that was doomed as it claimed the node: that compute
    /// again, verify first, with no walk, because a walk rooted at the
    /// node would visit its deps in an order the unbounded run did not.
    Compute,
    /// As `Compute`, for a compute the taint memo had sent straight to
    /// the closure.
    Direct,
}

/// A discarded frame that stands for the unbounded run's frame at its
/// position until the driver truncates the stack below it.
#[derive(Clone, Debug)]
pub(crate) struct VFrame {
    pub(crate) node: u32,
    pub(crate) generation: u32,
    pub(crate) cidx: u32,
    pub(crate) low: u32,
    pub(crate) suspect: bool,
    pub(crate) spec: bool,
    /// The read the frame was blocked in, `slot | generation << 32`.
    pub(crate) reading: u64,
    /// Which of the frame's reads of that node it was blocked in, from 1.
    pub(crate) ordinal: u32,
    /// The reads the frames below were in when this frame ran, as a hash
    /// of their ordinals: a frame of the same node at the same position
    /// under another read of a frame below is another run.
    pub(crate) path: u64,
    /// The level of the entry whose run made it.
    pub(crate) maker: u32,
    /// For an episode's member run, the episode, which waits in the
    /// context until this frame pops.
    pub(crate) ep: Option<u64>,
    /// The claims its run deferred before the doom, as `(slot, anchor)`.
    /// The discard released them, but in the unbounded run they are still
    /// held, so a cycle closed above counts them among its participants.
    pub(crate) deferred: Vec<(u32, u32)>,
}

/// What an entry's run left for the frame below the entry's level, whose
/// retry gets it at its read of the entry's node, as the unbounded run's
/// frame gets it at that read.
pub(crate) struct Delivery {
    pub(crate) node: u32,
    pub(crate) generation: u32,
    /// The frame's position: a run of the node elsewhere on the stack is
    /// another frame, which the unbounded run never blocked.
    pub(crate) cidx: u32,
    pub(crate) reading: u64,
    /// Which of the frame's reads of that node the delivery is for, from 1:
    /// a later read of the node in a retry is a read of its own.
    pub(crate) ordinal: u32,
    /// The frame's [`VFrame::path`].
    pub(crate) path: u64,
    /// The level of the entry whose retry recomputes the frame; the
    /// delivery is dropped when that run ends without reaching the read.
    pub(crate) maker: u32,
    /// The frame's current retry has read it: a later read of the node in
    /// the same retry is an ordinary one, as in the unbounded run.
    pub(crate) taken: bool,
    /// The outcome of the read the frame was blocked in, which a later
    /// outcome of the same read replaces; otherwise an earlier read's.
    pub(crate) blocked: bool,
    pub(crate) what: Delivered,
}

pub(crate) enum Delivered {
    /// An unwind out of a non-speculative entry: a cycle unwind, whose
    /// payload is the `CycleUnwind`, or a user panic.
    Unwind {
        cycle: bool,
        payload: Box<dyn Any + Send>,
    },
    /// The entry landed without a Clean value. The unbounded run's read
    /// takes the slot as the compute left it, poisoned, where a later read
    /// would walk and run the node again. It is that value, cloned when
    /// the delivery is made, because a later run in the context may
    /// compute the node again before the retry reads it.
    Landed(Box<dyn Any + Send>),
    /// The entry, at this node and generation, landed other than Clean in
    /// the walk of the frame's blocked read, which visited it once: the
    /// retry's walk of that read skips it, where a later read would run
    /// it again.
    Walked(u32, u32),
    /// The read ran its node's compute, a virtual frame above the frame's,
    /// which the retry runs again as that compute: a walk would first
    /// visit deps the replay of that frame's run has left Failed or Dirty.
    Direct,
}

pub(crate) struct Context {
    pub(crate) id: u64,
    pub(crate) rt: RuntimeId,
    /// The position of level 0.
    pub(crate) boundary: u32,
    pub(crate) levels: Vec<VecDeque<Entry>>,
    pub(crate) vstack: Vec<VFrame>,
    /// Each node's `vstack` positions, lowest first, with the generation
    /// its virtual frame recorded.
    pub(crate) vindex: HashMap<u32, Vec<(u32, u32)>>,
    /// The level of the entry the driver is running, or `None` until the
    /// loop starts.
    pub(crate) running: Option<u32>,
    pub(crate) pending: Vec<Delivery>,
    pub(crate) running_entry: Option<Entry>,
    /// Every entry of this context is speculative: the context was opened
    /// inside a speculative entry's run.
    pub(crate) spec_base: bool,
    /// The claims virtual hits took while an entry ran, anchored below its
    /// level, as `(level, slot)`: an entry that unwinds or is dropped
    /// releases them, since no frame of its run does.
    pub(crate) claims: Vec<(u32, u32)>,
    /// Episodes suspended by a doom of a member run, until that run's
    /// virtual frame pops.
    pub(crate) episodes: Vec<Episode>,
    /// The memo entries of frames this context's marking doomed, put back
    /// before the next entry runs, as a frame's entries outlive the runs
    /// nested in it.
    pub(crate) memo: taint_memo::Parked,
    /// The `Walked` deliveries taken, as the read they were for, the
    /// node and its generation, until the walk of that read skips them.
    pub(crate) walk_skip: Vec<(u64, u32, u32)>,
    /// The node and generation a `Direct` delivery just taken sends to
    /// its compute, until the read's walk does.
    pub(crate) direct: Option<(u32, u32)>,
}

impl Context {
    pub(crate) fn new(id: u64, rt: RuntimeId, boundary: u32, target: Entry) -> Self {
        Self {
            id,
            rt,
            boundary,
            levels: vec![VecDeque::from([target])],
            vstack: Vec::new(),
            vindex: HashMap::new(),
            running: None,
            running_entry: None,
            pending: Vec::new(),
            spec_base: target.spec,
            claims: Vec::new(),
            episodes: Vec::new(),
            memo: Vec::new(),
            walk_skip: Vec::new(),
            direct: None,
        }
    }

    /// The highest level holding an entry, and its oldest entry.
    pub(crate) fn next_entry(&self) -> Option<(u32, Entry)> {
        self.levels
            .iter()
            .enumerate()
            .rev()
            .find_map(|(level, q)| q.front().map(|e| (level as u32, *e)))
    }

    pub(crate) fn push_entry(&mut self, level: u32, entry: Entry) {
        let level = level as usize;
        if self.levels.len() <= level {
            self.levels.resize_with(level + 1, VecDeque::new);
        }
        self.levels[level].push_back(entry);
    }

    /// Removes the oldest entry at `level`, the one the driver ran.
    pub(crate) fn pop_entry(&mut self, level: u32) {
        let q = &mut self.levels[level as usize];
        q.pop_front();
        if q.is_empty() && level as usize + 1 == self.levels.len() && level > 0 {
            self.levels.pop();
        }
    }

    pub(crate) fn push_vframe(&mut self, frame: VFrame) {
        let pos = frame.cidx - self.boundary;
        debug_assert_eq!(
            pos as usize,
            self.vstack.len(),
            "incr-core: virtual frames are contiguous"
        );
        self.vindex
            .entry(frame.node)
            .or_default()
            .push((pos, frame.generation));
        self.vstack.push(frame);
    }

    /// Pops the top virtual frame.
    /// Delivers `what` to the read virtual frame `below` was blocked in.
    pub(crate) fn deliver_blocked(&mut self, below: &VFrame, what: Delivered) {
        self.pending.retain(|d| {
            !(d.blocked
                && d.node == below.node
                && d.generation == below.generation
                && d.cidx == below.cidx
                && d.reading == below.reading
                && d.ordinal == below.ordinal
                && d.path == below.path)
        });
        self.pending.push(Delivery {
            node: below.node,
            generation: below.generation,
            cidx: below.cidx,
            reading: below.reading,
            ordinal: below.ordinal,
            path: below.path,
            maker: below.maker,
            taken: false,
            blocked: true,
            what,
        });
    }

    /// Sends the read virtual frame `below` was blocked in to its compute
    /// in the retry of the entry at `level`.
    pub(crate) fn deliver_direct(&mut self, below: &VFrame, level: u32) {
        let same = |d: &Delivery| {
            matches!(d.what, Delivered::Direct)
                && d.node == below.node
                && d.generation == below.generation
                && d.cidx == below.cidx
                && d.reading == below.reading
                && d.ordinal == below.ordinal
                && d.path == below.path
        };
        if !self.pending.iter().any(same) {
            self.pending.push(Delivery {
                node: below.node,
                generation: below.generation,
                cidx: below.cidx,
                reading: below.reading,
                ordinal: below.ordinal,
                path: below.path,
                maker: level,
                taken: false,
                blocked: false,
                what: Delivered::Direct,
            });
        }
    }

    pub(crate) fn pop_vframe(&mut self) -> Option<VFrame> {
        let frame = self.vstack.pop()?;
        // As a frame's pop does: an entry it owns must not reach a run
        // below it, which the unbounded run makes after the pop.
        if value_code::thread_word() & value_code::MEMO_BIT != 0 {
            taint_memo::clear_from(frame.cidx + 1);
        }
        if let Some(stack) = self.vindex.get_mut(&frame.node) {
            stack.pop();
            if stack.is_empty() {
                self.vindex.remove(&frame.node);
            }
        }
        Some(frame)
    }

    /// The highest `vstack` position of `node` at `generation`.
    pub(crate) fn position(&self, node: u32, generation: u32) -> Option<u32> {
        let stack = self.vindex.get(&node)?;
        let &(pos, g) = stack.last()?;
        (g == generation).then_some(pos)
    }
}

#[cfg(not(loom))]
thread_local! {
    static CONTEXTS: RefCell<Vec<Context>> = const { RefCell::new(Vec::new()) };
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
    static SPECULATIVE: Cell<u32> = const { Cell::new(0) };
    static MEMBER_RUNS: RefCell<Vec<(u32, (u64, u32))>> = const { RefCell::new(Vec::new()) };
}

#[cfg(loom)]
loom::thread_local! {
    static CONTEXTS: RefCell<Vec<Context>> = RefCell::new(Vec::new());
    static NEXT_ID: Cell<u64> = Cell::new(1);
    static SPECULATIVE: Cell<u32> = Cell::new(0);
    static MEMBER_RUNS: RefCell<Vec<(u32, (u64, u32))>> = RefCell::new(Vec::new());
}

/// The id the next context or episode gets.
pub(crate) fn peek_id() -> u64 {
    NEXT_ID.with(Cell::get)
}

/// The path of a frame above one at `path` that is in its `ordinal`-th
/// read of a node.
pub(crate) fn extend_path(path: u64, ordinal: u32) -> u64 {
    path.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(u64::from(ordinal) + 1)
}

/// A fresh id for a context or an episode.
pub(crate) fn next_id() -> u64 {
    NEXT_ID.with(|n| {
        let id = n.get();
        n.set(id + 1);
        id
    })
}

/// Records that episode `ep`, as its id and root, runs a member at
/// `cidx`, for a doom that marks the run.
pub(crate) fn enter_member_run(cidx: u32, ep: (u64, u32)) {
    MEMBER_RUNS.with(|m| m.borrow_mut().push((cidx, ep)));
}

pub(crate) fn leave_member_run() {
    MEMBER_RUNS.with(|m| m.borrow_mut().pop());
}

/// The episode whose member run sits at `cidx`, if one does.
pub(crate) fn member_run_episode(cidx: u32) -> Option<u64> {
    MEMBER_RUNS.with(|m| {
        m.borrow()
            .iter()
            .rev()
            .find(|&&(c, _)| c == cidx)
            .map(|&(_, (ep, _))| ep)
    })
}

/// The root of the episode whose member run sits at `cidx`, if one does.
pub(crate) fn member_run_root(cidx: u32) -> Option<u32> {
    MEMBER_RUNS.with(|m| {
        m.borrow()
            .iter()
            .rev()
            .find(|&&(c, _)| c == cidx)
            .map(|&(_, (_, root))| root)
    })
}

/// Stores episode `ep` in the context of `rt` that holds its member run's
/// virtual frame.
pub(crate) fn suspend(rt: RuntimeId, ep: Box<Episode>) {
    let unplaced = with(|c| {
        match c
            .iter_mut()
            .rev()
            .find(|x| x.rt == rt && x.vstack.iter().any(|v| v.ep == Some(ep.id)))
        {
            Some(x) => {
                x.episodes.push(*ep);
                None
            }
            None => Some(ep),
        }
    });
    debug_assert!(
        unplaced.is_none(),
        "incr-core: a suspended episode has a virtual frame"
    );
    drop(unplaced);
}

/// The claims of the episodes of runtime `rt` that wait in a context.
pub(crate) fn suspended_members(rt: RuntimeId) -> Vec<u32> {
    with(|c| {
        c.iter()
            .filter(|x| x.rt == rt)
            .flat_map(|x| x.episodes.iter())
            .flat_map(|e| e.members.iter().copied().chain([e.root.0]))
            .collect()
    })
}

/// Takes episode `ep` out of context `id`.
pub(crate) fn take_episode(id: u64, ep: u64) -> Option<Box<Episode>> {
    with(|c| {
        let x = c.iter_mut().find(|x| x.id == id)?;
        let at = x.episodes.iter().position(|e| e.id == ep)?;
        Some(Box::new(x.episodes.remove(at)))
    })
}

/// Whether a speculative entry's run is in progress on this thread.
pub(crate) fn speculative() -> bool {
    SPECULATIVE.with(Cell::get) != 0
}

/// Counts a speculative entry's run while it lives; dropping it, by an
/// unwind included, gives the count back.
pub(crate) struct Speculating(());

impl Speculating {
    pub(crate) fn enter() -> Self {
        SPECULATIVE.with(|s| s.set(s.get() + 1));
        Self(())
    }
}

impl Drop for Speculating {
    fn drop(&mut self) {
        SPECULATIVE.with(|s| s.set(s.get() - 1));
    }
}

/// Runs `f` on the thread's contexts. `f` must not call user code or the
/// engine.
pub(crate) fn with<R>(f: impl FnOnce(&mut Vec<Context>) -> R) -> R {
    CONTEXTS.with(|c| {
        let mut c = c.borrow_mut();
        let r = f(&mut c);
        value_code::set_replay_bit(!c.is_empty());
        r
    })
}

/// Opens a context, returning its id.
pub(crate) fn open(rt: RuntimeId, boundary: u32, target: Entry) -> u64 {
    let id = next_id();
    with(|c| c.push(Context::new(id, rt, boundary, target)));
    id
}

/// Removes context `id`, if it is still open.
pub(crate) fn close(id: u64) {
    let removed = with(|c| c.iter().position(|x| x.id == id).map(|at| c.remove(at)));
    drop(removed);
}

/// Removes every context of `rt` whose boundary is at or above `cidx`:
/// a caught unwind has left the reads that opened them.
pub(crate) fn drop_from(rt: RuntimeId, cidx: u32) {
    remove_where(|x| x.rt == rt && x.boundary >= cidx);
}

/// Removes the contexts of `rt` that the unwind of the frame at `cidx`
/// leaves: those opened by its reads, above it, and one not yet running
/// whose target is that frame, since the read that would run it is
/// unwinding too. A running context whose entry that frame is stays, for
/// its driver's catch.
pub(crate) fn unwind_past(rt: RuntimeId, cidx: u32) {
    remove_where(|x| {
        x.rt == rt && (x.boundary > cidx || (x.boundary == cidx && x.running.is_none()))
    });
}

/// [`unwind_past`] for a member run at `cidx`, which sits at its anchor
/// below frames that are still live: only contexts opened since the run
/// began, with ids from `first`, are its reads'.
pub(crate) fn unwind_past_since(rt: RuntimeId, cidx: u32, first: u64) {
    remove_where(|x| {
        x.rt == rt
            && x.id >= first
            && (x.boundary > cidx || (x.boundary == cidx && x.running.is_none()))
    });
}

fn remove_where(mut gone: impl FnMut(&Context) -> bool) {
    if value_code::thread_word() & value_code::REPLAY_BIT == 0 {
        return;
    }
    let removed: Vec<Context> = with(|c| {
        let mut out = Vec::new();
        let mut i = 0;
        while i < c.len() {
            if gone(&c[i]) {
                out.push(c.remove(i));
            } else {
                i += 1;
            }
        }
        out
    });
    drop(removed);
}

/// Removes every context on the thread: an outermost engine entry
/// starts with none.
pub(crate) fn reset() {
    if value_code::thread_word() & value_code::REPLAY_BIT == 0 {
        return;
    }
    let removed = with(std::mem::take);
    drop(removed);
}

/// The innermost context of `rt` whose boundary is at or below `cidx`,
/// as an index into the stack.
pub(crate) fn covering(c: &[Context], rt: RuntimeId, cidx: u32) -> Option<usize> {
    c.iter().rposition(|x| x.rt == rt && x.boundary <= cidx)
}

/// A run at position `cidx` that is not a driver entry's, a member run,
/// ended. The deliveries for the frames its run made are read again by its
/// retry if a doom discarded it, and dropped otherwise, as the driver does
/// for an entry's run.
pub(crate) fn run_ended(rt: RuntimeId, cidx: u32, doomed: bool) {
    let ended: Vec<Delivery> = with(|c| {
        let Some(at) = covering(c, rt, cidx) else {
            return Vec::new();
        };
        let x = &mut c[at];
        let level = cidx - x.boundary;
        if doomed {
            for d in x.pending.iter_mut().filter(|d| d.maker == level) {
                d.taken = false;
            }
            return Vec::new();
        }
        x.pending.extract_if(.., |d| d.maker == level).collect()
    });
    drop(ended);
}

/// The innermost context of `rt`, as an index into the stack.
pub(crate) fn innermost(c: &[Context], rt: RuntimeId) -> Option<usize> {
    c.iter().rposition(|x| x.rt == rt)
}
