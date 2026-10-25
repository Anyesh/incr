//! `Runtime<C>`: the strategy-parameterized incremental computation engine.
//!
//! Single struct, single `impl` block, monomorphized at compile time into
//! the single-threaded variant (`Runtime<Local>`) and the concurrent
//! variant (`Runtime<Shared>`). The user-facing crates `incr-compute` and
//! `incr-concurrent` re-export the appropriate alias.
//!
//! This first slice ships the core algorithm:
//! - `create_input<T>` / `create_query<T, F>`: node construction.
//! - `get<T>(handle)` / `set<T>(handle, value)`: the user-facing API.
//! - `ensure_clean`: iterative post-order walker that recomputes dirty
//!   nodes in topological order.
//! - `run_compute`: claim Computing, run the closure, observe new deps,
//!   update edges, Release Clean. Includes red/green early cutoff.
//! - `mark_dependents_dirty`: BFS dirty walk from a mutated input.
//!
//! Deferred to follow-ups: handle validation (runtime_id + generation
//! checks), introspection (graph_snapshot, labels), real tracing,
//! collection operators, soundness fixes (race-detection ordering with
//! AcqRel, overflow-dep reclamation).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use crate::arena_registry::{ArenaRegistry, ArenaTags};
use crate::cells::Cells;
use crate::children::ChildIndex;
use crate::cycle::{
    self, ComputeCtl, Cycle, CycleStrategy, CycleUnwind, ErasedOpts, ErasedStrategy, Outcome,
    QueryOpts,
};
use crate::cycle_state::{self, EntryDepth, UnwindKind};
use crate::deferred::{Deferred, DeleteBag, NodeDrops};
use crate::dep_set::{self, DepDiff, DepHashSet, DepLookup};
use crate::dep_stack::{ActiveRead, DepStack};
use crate::dependent_list::DependentList;
use crate::fx::FxBuildHasher;
use crate::generic_arena::GenericArena;
use crate::handle::{ErasedIncr, Incr, RuntimeId};
use crate::lock_order::LockKind;
use crate::locks::OrderedLock;
use crate::node::{NodeData, NodeId, RetiredDeps};
use crate::reclaim_probe::{self, RetireSite};
use crate::replay;
use crate::reserve;
use crate::segmented_nodes::{
    SegmentedNodes, HAS_CHILDREN, HAS_INDEX, HAS_STRATEGY, IN_GROUP, NOT_REPLAYABLE, OWNED, WATCHED,
};
use crate::set_stack;
use crate::slot_vec::SlotVec;
use crate::state::{self, NodeState};
use crate::table::{NodeTable, TableKey};
use crate::taint_memo;
use crate::value::Value;
use crate::value_code;
use crate::value_slot::{Compared, Displaced};
use crate::waits;

/// Compute closure: takes a borrow of the runtime, the node's slot and
/// the frame's control block, which says whether this is a recompute
/// (whose write compares first) and whether the value must be buffered
/// instead of written, and which records the write. Returns what it did
/// with the value: `Unchanged` is the early cutoff.
pub(crate) type ComputeFn<C> =
    Box<dyn Fn(&Runtime<C>, u32, &ComputeCtl) -> Outcome + Send + Sync + 'static>;

/// Per-runtime mutable state guarded by the inner lock. Holds everything
/// that's not on the per-node `NodeData` and not in an arena.
pub(crate) struct Inner<C: Cells> {
    pub(crate) compute_fns: SlotVec<Option<ComputeFn<C>>>,
    pub(crate) dependents: SlotVec<DependentList>,
    pub(crate) arena_tags: ArenaTags,
    pub(crate) labels: HashMap<u32, String>,
    pub(crate) trace_log: Vec<crate::trace::NodeTrace>,
    /// Node slots released by delete_node, awaiting recycling.
    pub(crate) free_nodes: Vec<u32>,
    pub(crate) observers: Observers<C>,
    /// Nodes an operator created for its own output and adopted: it sets
    /// them but never reads them, so no reverse edge records them.
    /// Sparse, because only operators like `group_by` own nodes. Kept in
    /// step with `owner_of` and unhooked by the child's delete under the
    /// same write hold as its generation bump, so an entry always names
    /// the node that was adopted.
    pub(crate) owned: HashMap<u32, Vec<NodeId>>,
    pub(crate) owner_of: HashMap<u32, NodeId>,
    /// The child index of each creator, flagged by `HAS_INDEX`. Boxed so
    /// that a running creator's frame can point at it while the map
    /// moves.
    pub(crate) children: HashMap<u32, Box<ChildIndex<C>>, FxBuildHasher>,
    /// The cycle strategies of the queries that have one, flagged by
    /// `HAS_STRATEGY`.
    pub(crate) cycle_strategies: HashMap<u32, Arc<dyn ErasedStrategy>>,
    /// The group each landed cycle member belongs to, flagged by
    /// `IN_GROUP`.
    pub(crate) member_recs: HashMap<u32, Arc<GroupInfo>>,
}

/// A landed cycle group: its members with the generations they landed
/// under, sorted by slot, and the cycle readers see.
pub(crate) struct GroupInfo {
    pub(crate) members: Vec<(u32, u32)>,
    pub(crate) cycle: Cycle,
}

impl<C: Cells> Inner<C> {
    fn new() -> Self {
        Self {
            compute_fns: SlotVec::new(),
            dependents: SlotVec::new(),
            arena_tags: ArenaTags::default(),
            labels: HashMap::new(),
            trace_log: Vec::new(),
            free_nodes: Vec::new(),
            observers: Observers::new(),
            owned: HashMap::new(),
            owner_of: HashMap::new(),
            children: HashMap::default(),
            cycle_strategies: HashMap::new(),
            member_recs: HashMap::new(),
        }
    }
}

/// Token returned by [`Runtime::observe`]; pass to [`Runtime::unobserve`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ObserverId(u64);

/// Named points in the protocol where a race test can park one thread
/// while another acts. Outside `cfg(test)` [`Runtime::park`] is an empty
/// inline function, so these compile away.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ParkPoint {
    /// In `ensure_clean`, just before `compute_one` on a visited node, or
    /// before the walk waits on a node another thread is computing.
    BeforeComputeOne(NodeId),
    /// In `compute_one`, after the state load and before the claim CAS.
    BeforeClaim(NodeId),
    /// In a read's walk loop, after the walk and before the slot read.
    AfterWalk(NodeId),
    /// In `compute_attempts`, after the claim CAS and before the check for
    /// another thread's doom reservation.
    BeforeReservationCheck(NodeId),
    /// In `compute_one`, after a walk entry's claim and before the check
    /// for episodes other threads landed since the walk planned.
    AfterWalkClaim(NodeId),
    /// In red/green verification, after the dep scan found no change and
    /// before the landing CAS.
    VerifyAfterScan(NodeId),
    /// In `stash`, under the stash lock, after the revision check and
    /// before the entry and its count are stored.
    StashAfterCheck(NodeId),
    /// In `set`, after the runtime check and before the claim's hold.
    SetBeforeClaim(NodeId),
    /// In `set`, under its claim, with no guard held, before the compare.
    SetAfterClaim(NodeId),
    /// In `set`, after `bump_revision` and before the dirty walk.
    SetAfterBump,
    /// In `set`, after the dirty walk and before the revision is recorded
    /// as walked.
    SetAfterWalk,
    /// In a delete, after the dependents check and the `Dead` claim, with
    /// no guard held.
    DeleteAfterCheck(NodeId),
    /// In a delete, on finding the node claimed `Dead` by another delete
    /// that has not bumped its generation yet.
    DeleteWaitsOnClaim(NodeId),
    /// In a strict delete, after the handle check and before any read hold.
    DeleteAfterHandleCheck(NodeId),
    /// In `claim_set` for a delete of root `NodeId`'s owned set, under
    /// the write hold, on finding a busy member and before rolling the
    /// claimed members back.
    ClaimSetBeforeRollback(NodeId),
    /// In a strict delete of root `NodeId`'s owned set, after `claim_set`
    /// rolled back around a busy member and released its hold, before
    /// the wait on that member.
    ClaimSetRolledBack(NodeId),
    /// In `unwatch`, under its write hold, after `WATCHED` is cleared and
    /// before the node's entry is dequeued.
    UnwatchBeforeDequeue(NodeId),
    /// In `get`, after the handle's runtime check and before its
    /// generation is compared.
    GetAfterHandleCheck(NodeId),
    /// In `get`, after the node was found Clean and before its value is
    /// read from the arena.
    GetBeforeRead(NodeId),
    /// In `get`'s value-code arm, after the fast path missed and before
    /// the state is loaded again.
    ValueCodeBeforeReload(NodeId),
    /// In a collection operator's rebuild, just after it read an
    /// upstream's snapshot and released that upstream's guard.
    RebuildAfterSnapshot,
    /// In `compute_one`, after a Stale publish landed `New` and before
    /// the retry, with the attempt's bag carried along.
    StaleRetry(NodeId),
    /// In `compute_one`, after an episode landed the root, before its
    /// reader learns how.
    EpisodeLanded(NodeId),
    /// At a doom, after the node the read wanted was released and pushed
    /// and before the reading frame gets its placeholder.
    AfterDoom(NodeId),
    /// In a doomed frame's read of `NodeId`, before its state is loaded.
    DoomedRead(NodeId),
    /// In a replay driver's loop, after it took its next entry and before
    /// the virtual frames at and above that entry's level pop.
    BeforeEntry(NodeId),
    /// In `hold_landing`, before it loads the state of `NodeId`, which a
    /// replay entry's run just landed.
    HoldLanding(NodeId),
    /// In `compute_one`, on finding the node claimed by another thread,
    /// before its first wait on that claim.
    ComputeWaits(NodeId),
    /// In group verify's step 1, after it claimed member `NodeId`.
    GroupMemberClaimed(NodeId),
    /// In a read's judgement of a node with a virtual frame, after the scan
    /// of the virtual frames above it and before their claims.
    BeforeVirtualClaim(NodeId),
    /// In a Failed read that found a payload it may take, after it loaded
    /// the node's generation and before the record lock.
    FailedReadBeforeRecheck(NodeId),
    /// In `seed_query`'s `inner.write()`, before the trust check of the
    /// trusted dep `NodeId`.
    SeedBeforeTrust(NodeId),
    /// In `seed_query`'s `inner.write()`, after the trust check of the
    /// trusted dep `NodeId`.
    SeedAfterTrust(NodeId),
    /// In `inspect`, after its first read hold and before the clone.
    InspectBetweenHolds(NodeId),
    /// In `stabilize`, after it took its observers and before the first
    /// one runs.
    StabilizeAfterDrain,
    /// In `observer_read`, after the entry generation compare and before
    /// `ensure_clean`.
    ObserverAfterEntryCheck(NodeId),
    /// In `observer_read`, after `ensure_clean` returned and before the
    /// generation compare that follows it.
    ObserverAfterEnsureClean(NodeId),
    /// In `observer_read`, after the compare that follows `ensure_clean`
    /// and before the seqlock pre-check.
    ObserverBeforePreCheck(NodeId),
}

/// Ends the whole test binary: a parked thread's panic can be caught as a
/// compute's, and a test that accepts any error then passes on it.
#[cfg(test)]
pub(crate) fn park_timed_out(test: Option<&str>) -> ! {
    use std::io::Write as _;
    // Past the test harness's output capture, which the abort discards.
    let _ = writeln!(
        std::io::stderr(),
        "incr-core test: a thread parked by {} was never released",
        test.unwrap_or("an unnamed thread")
    );
    std::process::abort()
}

/// The panic of a read that found its node `Dead` under the handle's
/// generation: a delete has claimed it and not yet unhooked it.
#[cold]
#[inline(never)]
fn deleted_while_reading(id: NodeId) -> ! {
    panic!(
        "incr-core: node {} was deleted while a read was reaching it (a \
         compute raced delete_node on one of its recorded dependencies)",
        id.0,
    );
}

/// What [`Runtime::delete_erased`] did with a node.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum DeleteOutcome {
    Deleted,
    /// The handle's node was already deleted, by this caller earlier or
    /// by a concurrent delete that has now finished.
    AlreadyGone,
    /// The node has a dependent. Nothing changed.
    HasDependents,
}

/// Why [`Runtime::try_delete_node`] deleted nothing.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeleteError {
    /// The node, or a node it owns, has a dependent: a query that
    /// computed against it. Delete the dependents first, or use
    /// [`Runtime::delete_with_dependents`].
    HasDependents,
    /// The handle names no live node of this runtime: the node was
    /// deleted, possibly by a concurrent delete, or the handle comes from
    /// another runtime. These are the handles [`Runtime::is_alive`]
    /// reports false for.
    NotAlive,
}

impl fmt::Display for DeleteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DeleteError::HasDependents => "the node or a node it owns still has dependents",
            DeleteError::NotAlive => "the handle names no live node of this runtime",
        })
    }
}

impl std::error::Error for DeleteError {}

/// Why a strict delete refused. Nothing changed.
enum StrictRefusal {
    /// The node was already deleted, or a concurrent delete finished it.
    Gone,
    /// The node at this slot, the root or one it owns, has a dependent.
    HasDependents(u32),
}

/// A node a delete claimed `Dead` and cleared under that claim, waiting
/// for the write hold that unhooks and recycles it.
struct Retired<C: Cells> {
    handle: ErasedIncr,
    old_deps: Vec<NodeId>,
    arena_slot: u32,
    type_tag: u16,
    dep_list: Option<RetiredDeps<C>>,
}

/// What the single-node claim of a delete found.
enum SingleClaim {
    /// The node is `Dead` and this delete's.
    Claimed,
    /// The node was already deleted, or a concurrent delete finished it.
    Gone,
    HasDependents,
    /// The node owns nodes, so only a delete of its owned set can take it.
    Owner,
}

/// Records `rev` as walked when dropped. A drop guard rather than a call
/// at the end of `set`, because an unwind that skipped the completion
/// would stall the watermark, and with it every later `set`, for good.
struct WalkDone<'a, C: Cells> {
    rt: &'a Runtime<C>,
    rev: u64,
}

impl<C: Cells> Drop for WalkDone<'_, C> {
    fn drop(&mut self) {
        C::complete_walk(&self.rt.walk, self.rev);
    }
}

/// `set`'s claim on its input ([`NodeData::SET_CLAIM`]), from the claim
/// to its release, with the claim's set-stack entry. Dropped while
/// `held` (an unwind before the write, a `PartialEq` panic, or an exit
/// that writes nothing), it releases the claim over the stamp it found
/// and pops the entry. The write's exit is [`Self::finish_written`].
struct SetClaim<'a, C: Cells> {
    rt: &'a Runtime<C>,
    node: &'a NodeData<C>,
    flags: &'a C::U8,
    id: NodeId,
    /// The input's `verified_at` the claim was taken over.
    base: u64,
    held: bool,
}

impl<C: Cells> SetClaim<'_, C> {
    /// The bump, the settle, the release and the walk, none of which
    /// runs user code, so the written value is never left unpublished.
    /// `held` is cleared first so that an unwind out of this never
    /// releases the claim a second time. The release precedes the walk
    /// so that a delete or set waiting on the claim is not held up by it.
    #[inline(always)]
    fn finish_written(mut self) -> u64 {
        self.held = false;
        let rev = self.rt.bump_revision();
        let walk_done = WalkDone { rt: self.rt, rev };
        self.rt.park(ParkPoint::SetAfterBump);
        let stamp = NodeData::<C>::input_stamp(rev);
        self.node.settle_changed_at(stamp);
        self.node.release_set_claim(self.base, rev);
        set_stack::pop(self.rt.runtime_id, self.id.0);
        self.rt.mark_dependents_dirty(self.id, self.flags);
        self.rt.park(ParkPoint::SetAfterWalk);
        drop(walk_done);
        self.rt.drop_released_tickets();
        // Pairs with the fence in `stash`, after its store of the count.
        C::fence_seqcst();
        if C::u32_load_relaxed(&self.rt.stash_count) != 0 {
            self.rt.drop_stashes();
        }
        std::mem::forget(self);
        stamp
    }
}

impl<C: Cells> Drop for SetClaim<'_, C> {
    fn drop(&mut self) {
        if self.held {
            self.node.release_set_claim(self.base, self.base);
            set_stack::pop(self.rt.runtime_id, self.id.0);
        }
    }
}

/// What `publish_deps` found when it validated a compute's recorded deps.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Publish {
    /// Edges are current; the compute may land `Clean`.
    Published,
    /// A newly added dep was not observed `Clean` at or below the
    /// walked watermark captured at the claim, so a concurrent change may
    /// have missed this node, which had no reverse edge yet (B3). Edges are
    /// published; the compute must land `Dirty`.
    Superseded,
    /// A recorded dep was deleted, and possibly recycled, after the read
    /// (B2). Nothing was published.
    Stale,
}

#[cfg(test)]
pub(crate) type ParkHook = Box<dyn Fn(ParkPoint) + Send + Sync>;

/// Test hook that hands an item to the bag carried past a park point, so
/// a test can put a counting item where no real path puts one yet.
#[cfg(test)]
pub(crate) type BagHook =
    Box<dyn Fn(ParkPoint) -> Option<crate::deferred::ErasedDrop> + Send + Sync>;

/// First visits after which an `ensure_clean` walk switches to exact mode.
const EXACT_MODE_AT: u32 = 4096;

/// How `compute_one` treats the node it was handed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Visit {
    /// A walk root or an untainted walk entry: verify, and recompute if
    /// that fails.
    Full,
    /// A tainted walk entry (`Walk::taint`): verify, and if that fails
    /// leave the node Dirty for the root's own compute to re-read. The
    /// payload is the depth of the frame that owns the taint.
    VerifyOnly(u32),
    /// A root the taint memo holds: claim and run the closure without
    /// verifying, because its verify already failed under taint.
    Direct,
}

/// What a walk under `Shared` planned under, as `Runtime::planned`.
#[derive(Copy, Clone)]
struct Planned {
    all: u64,
    own: u64,
}

/// Test toggle: with `INCR_FORCE_REVISIT` set, every untainted walk
/// entry under `Shared` counts once as planned New, so a claim of it from
/// Dirty visits it again.
#[cfg(test)]
static FORCE_REVISIT: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("INCR_FORCE_REVISIT").is_some());

/// Test toggle: with `INCR_FORCE_REPLAN` set, every walk entry's check
/// finds another thread's episode.
#[cfg(test)]
static FORCE_REPLAN: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var_os("INCR_FORCE_REPLAN").is_some());

#[cfg(test)]
thread_local! {
    /// `FORCE_REPLAN` for this thread alone.
    pub(crate) static FORCE_REPLAN_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// How `compute_one` left the node it was handed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Computed {
    /// The node is as settled as the visit allows.
    Done,
    /// As `Done`, but a closure run left the node not Clean, so a walk
    /// that computes it again runs the closure again.
    Unsettled,
    /// The slot's generation moved during the call: the node was deleted
    /// and its slot recycled, so the closure read under the hold belongs
    /// to a node that no longer exists. Any claim taken was released
    /// with the discard landing, and nothing ran against the occupant.
    Gone,
    /// The node read an open cycle: its claim moved to this thread's held
    /// map, its value is buffered there, and the root of the cycle lands
    /// it.
    Deferred,
    /// The read that asked for the node was past its budget, so the node
    /// was released unrun and pushed as a replay entry, and the reading
    /// frame is doomed.
    Doomed,
    /// The node's run was doomed and discarded: it is a virtual frame,
    /// and a replay driver runs it again.
    Discarded,
    /// Another thread landed an episode since the walk planned: the node
    /// was left unrun and unclaimed, and the walk prunes its plan.
    Replan,
    /// Another thread landed the node since the walk planned it: a claim
    /// wait ended with it Dirty or New, not released unrun, or the walk
    /// planned it New and claimed it from Dirty. The node was left unrun
    /// and unclaimed, and the walk visits it again, as a read that came
    /// after that landing expands its deps before claiming it.
    Revisit,
    /// A claim wait ended with the node New or Dirty and a virtual frame
    /// of it on this thread, which stands for a frame on the stack: the
    /// walk skips it and taints the path under this memo owner.
    Skipped(u32),
}

/// `clean_walk` mode bit: a compute about to run a closure dooms the
/// reading frame instead.
pub(crate) const ARMED: u8 = 1;
/// `clean_walk` mode bit: a replay context of this runtime is open, so
/// the walk skips nodes that have virtual frames.
pub(crate) const VIRT: u8 = 2;

/// Which read reached a slow path: `get` panics where `try_get` returns
/// `Err(Cycle)`.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReadKind {
    Get,
    TryGet,
}

/// `Entry::taint` of a visit entry.
const VISIT: u32 = u32::MAX;

/// `Entry::taint` of a visit entry whose node was last loaded Computing,
/// so the visit takes it as Computing without loading it again. A stale
/// hint costs one backoff, after which the state is always reloaded;
/// every load skipped is a branch point fewer in the loom models.
const VISIT_COMPUTING: u32 = u32::MAX - 1;

/// `Entry::taint` of a pending entry no skipped dep has reached.
const NO_TAINT: u32 = u32::MAX - 3;

/// `NO_TAINT` of a node the walk's visit found New. Above `NO_TAINT`, so
/// a taint merged into it drops the mark; merging `NO_TAINT` is skipped.
const NO_TAINT_NEW: u32 = u32::MAX - 2;

/// What a walk that hands `compute_one` an untainted entry planned of it.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Seen {
    /// No walk that can visit the node again.
    Unwalked,
    Dirty,
    New,
}

/// One entry of `ensure_clean`'s walk. A visit (`taint >= VISIT_COMPUTING`)
/// expands `x`'s non-Clean deps; the pending entry is pushed on the first
/// visit and runs `x`'s compute once they are done.
#[derive(Copy, Clone)]
struct Entry {
    id: NodeId,
    /// `VISIT` or `VISIT_COMPUTING`, or on a pending entry the lowest
    /// frame depth among the Computing deps this thread holds that were
    /// skipped under it, or `NO_TAINT` (`NO_TAINT_NEW` if its visit found
    /// it New). A tainted node is reachable only through edges that
    /// may be stale, so the walk must not run its closure: the parent's
    /// closure decides whether it is still read. The kind shares the
    /// field so that an entry stays 8 bytes.
    taint: u32,
}

impl Entry {
    #[inline(always)]
    fn is_pending(self) -> bool {
        self.taint < VISIT_COMPUTING
    }
}

/// The stack of `ensure_clean`'s post-order walk.
#[derive(Clone, Default)]
pub(crate) struct Walk {
    work: Vec<Entry>,
    first_visits: u32,
    /// Exact mode: every node with a `(x, true)` entry on `work`, and every
    /// node a compute left not Clean, kept from the switch on so a popped
    /// `(x, false)` of one of them is skipped instead of expanded again.
    /// `None` below the switch, which keeps the walk free of set
    /// operations for small graphs.
    visited: Option<HashSet<NodeId>>,
    /// Created at the walk's first New expansion or group expansion, in
    /// one box so that a walk with neither tests one pointer at its drop.
    expanded: Option<Box<Expanded>>,
}

#[derive(Default, Clone)]
struct Expanded {
    /// New nodes with installed deps the walk has expanded.
    new: HashSet<NodeId>,
    /// The cycle groups the walk has expanded whole.
    groups: Vec<Arc<GroupInfo>>,
}

impl Walk {
    /// Drops the pending entry of `id` the walk was to run next, whose
    /// compute ran elsewhere. The root's entry stays: a walk stopped at a
    /// non-root entry of the root's own node (a self-edge's repeat) still
    /// owes its caller the root's compute.
    pub(crate) fn ran(&mut self, id: u32) {
        if self.work.len() > 1
            && self
                .work
                .last()
                .is_some_and(|e| e.is_pending() && e.id.0 == id)
        {
            self.work.pop();
        }
    }
    #[inline]
    fn new(root: NodeId, root_state: NodeState) -> Self {
        Self {
            work: vec![Entry {
                id: root,
                taint: Self::visit_kind(root_state),
            }],
            first_visits: 0,
            visited: None,
            expanded: None,
        }
    }

    /// Whether this is the walk's first expansion of the New node `id`.
    #[cold]
    #[inline(never)]
    fn first_new_expansion(&mut self, id: NodeId) -> bool {
        self.expanded
            .get_or_insert_with(Box::default)
            .new
            .insert(id)
    }

    #[inline(always)]
    fn push(&mut self, entry: Entry) {
        if self.work.len() == self.work.capacity() {
            self.grow();
        }
        self.work.push(entry);
    }

    /// Grows through a fresh allocation, not `realloc`, because glibc's
    /// `realloc` bypasses its thread cache and costs several times a
    /// cached `malloc` and `free`. The first growth goes straight to 16
    /// entries, so a short walk grows once.
    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        let mut grown = Vec::with_capacity((self.work.capacity() * 2).max(16));
        grown.extend_from_slice(&self.work);
        self.work = grown;
    }

    #[inline]
    fn visit(&mut self, id: NodeId, new: bool) {
        self.push(Entry {
            id,
            taint: if new { NO_TAINT_NEW } else { NO_TAINT },
        });
        self.first_visits += 1;
        if self.visited.is_some() || self.first_visits == EXACT_MODE_AT {
            self.visit_exact(id);
        }
    }

    #[cold]
    #[inline(never)]
    fn visit_exact(&mut self, id: NodeId) {
        match &mut self.visited {
            Some(visited) => {
                visited.insert(id);
            }
            None => {
                self.drop_repeats(self.work.len() - 1);
                self.visited = Some(
                    self.work
                        .iter()
                        .filter(|e| e.is_pending())
                        .map(|e| e.id)
                        .collect(),
                );
            }
        }
    }

    /// At the switch, drops what the walk pushed below `top` since it first
    /// expanded a node that already had a pending entry: exact mode would
    /// have skipped that expansion, and each repeated pending entry of a
    /// node that does not land Clean would run its compute, nested walks
    /// included, once more. The dropped entries' taint goes to the pending
    /// entry below them, which their pops would have reached.
    fn drop_repeats(&mut self, top: usize) {
        let mut pending = HashSet::new();
        let Some(cut) = self.work[..top]
            .iter()
            .position(|e| e.is_pending() && !pending.insert(e.id))
        else {
            return;
        };
        let taint = self.work[cut..top]
            .iter()
            .filter(|e| e.is_pending())
            .map(|e| e.taint)
            .min()
            .unwrap_or(NO_TAINT);
        if taint < NO_TAINT {
            if let Some(parent) = self.work[..cut].iter_mut().rev().find(|e| e.is_pending()) {
                parent.taint = parent.taint.min(taint);
            }
        }
        self.work.drain(cut..top);
    }

    /// A compute of `id` the walk ran left it not Clean, so every other
    /// pending entry or visit of it would run its closure, nested walks
    /// included, once more: switches to exact mode now, with `id` visited
    /// and its pending entries dropped, their taint passed down as
    /// `drop_repeats` passes it. The root's entry stays for its caller.
    #[cold]
    #[inline(never)]
    fn unsettled(&mut self, id: NodeId) {
        // From the switch on, a node is expanded at most once per walk.
        if self.visited.is_some() {
            return;
        }
        self.drop_repeats(self.work.len());
        for at in (1..self.work.len()).rev() {
            let entry = self.work[at];
            if entry.is_pending() && entry.id == id {
                self.work.remove(at);
                if entry.taint < NO_TAINT {
                    if let Some(parent) = self.work[..at].iter_mut().rev().find(|e| e.is_pending())
                    {
                        parent.taint = parent.taint.min(entry.taint);
                    }
                }
            }
        }
        let mut visited: HashSet<NodeId> = self
            .work
            .iter()
            .filter(|e| e.is_pending())
            .map(|e| e.id)
            .collect();
        visited.insert(id);
        self.visited = Some(visited);
    }

    /// Drops the plan above `work[keep - 1]`, and with it the visits and
    /// expansions of the nodes it left no pending entry, so that a later
    /// visit of one plans it as a walk that never met it would.
    fn prune(&mut self, keep: usize) {
        let mut gone: HashSet<NodeId> = self
            .work
            .drain(keep..)
            .filter(|e| e.is_pending())
            .map(|e| e.id)
            .collect();
        for e in self.work.iter().filter(|e| e.is_pending()) {
            gone.remove(&e.id);
        }
        if !gone.is_empty() {
            self.forget(|id| gone.contains(id));
        }
    }

    /// Visits `id` again as a walk that never met it would, past a first
    /// visit that expanded it (or the group it is now a member of).
    #[cold]
    #[inline(never)]
    fn revisit(&mut self, id: NodeId) {
        self.forget(|&n| n == id);
        self.push(Entry { id, taint: VISIT });
    }

    fn forget(&mut self, gone: impl Fn(&NodeId) -> bool) {
        if let Some(visited) = &mut self.visited {
            visited.retain(|id| !gone(id));
        }
        if let Some(expanded) = &mut self.expanded {
            expanded.new.retain(|id| !gone(id));
            expanded
                .groups
                .retain(|g| !g.members.iter().any(|&(m, _)| gone(&NodeId(m))));
        }
    }

    #[inline]
    fn expand(&mut self, dep: NodeId, dep_state: NodeState) {
        self.push(Entry {
            id: dep,
            taint: Self::visit_kind(dep_state),
        });
    }

    #[inline(always)]
    fn visit_kind(observed: NodeState) -> u32 {
        // A table, so that the expansion of a dep costs one load and no
        // branch or compare.
        const KIND: [u32; 8] = {
            let mut kind = [VISIT; 8];
            kind[NodeState::Computing as usize] = VISIT_COMPUTING;
            kind[NodeState::ComputingDirty as usize] = VISIT_COMPUTING;
            kind
        };
        KIND[observed as usize]
    }

    #[inline]
    fn is_visited(&self, id: NodeId) -> bool {
        match &self.visited {
            Some(visited) => Self::contains(visited, id),
            None => false,
        }
    }

    #[cold]
    #[inline(never)]
    fn contains(visited: &HashSet<NodeId>, id: NodeId) -> bool {
        visited.contains(&id)
    }

    /// Taint the entry whose deps are being processed: the nearest
    /// pending entry from the top, because everything above it is a
    /// sibling visit or a finished subtree.
    fn taint(&mut self, owner: u32) {
        if let Some(parent) = self.work.iter_mut().rev().find(|e| e.is_pending()) {
            parent.taint = parent.taint.min(owner);
        }
    }
}

#[cfg(all(test, not(loom)))]
fn forced_revisit() -> bool {
    *FORCE_REVISIT
}

#[inline(always)]
fn is_computing(state: NodeState) -> bool {
    matches!(state, NodeState::Computing | NodeState::ComputingDirty)
}

#[cold]
#[inline(never)]
fn reached_under_own_set(id: NodeId) -> ! {
    panic!(
        "incr-core: node {} was reached while this thread's set of it was in progress",
        id.0,
    );
}

/// The runtime.
///
/// # Closures may run more than once
///
/// A query's closure can run several times for one change: a retry after
/// a dep was released under it, a retry after another thread wounded its
/// claim, a discarded run, a replayed run ([`Replay`](crate::Replay)) and each iteration
/// of a cycle's episode. So a closure must be pure, must terminate on
/// every input, and must not panic or loop on a placeholder
/// ([`Runtime::set_placeholder`]).
///
/// A replayed run can be speculative: after a read returned a
/// placeholder, the run may evaluate queries the run without a budget
/// would never evaluate. Such a run lands nothing: the value or panic it
/// ends with, and the children it created, are kept for the revision, and
/// a later read of that query takes them if nothing the run read has
/// changed, and otherwise runs the closure again. So does a later read of a
/// query whose run kept nothing: it read a dep that did not land Clean or
/// a panic that another speculative run kept, it read through a run that
/// kept nothing, or it met a cycle. A
/// [`Replay::Native`](crate::Replay::Native) query runs speculatively at
/// most once per revision.
///
/// # Panics in user code
///
/// A panic in a user `Hash`, `Eq`, `PartialEq`, `Clone`, `Serialize` or
/// closure leaves the runtime consistent: the node lands `Failed`, Dirty
/// or New, its edges are published, and no lock is held while user code
/// runs. The rule for a user `Drop` is std's own. One that panics on a
/// normal path propagates as an ordinary panic after the operation has
/// landed (a value or dep list displaced by a compute is dropped at the
/// compute's exit, a deleted node's closure, observers and value at the
/// delete's exit). One that panics on an exit already carrying a panic
/// is swallowed, so the original panic arrives unchanged. One that panics
/// while the thread is already unwinding aborts the process, exactly as a
/// panicking `Drop` inside a `Vec` does.
///
/// Under a budget, replay also builds values that the run without one
/// never builds: a doomed run's value, the value of a speculative run that
/// no later read takes, and the clones a speculative run's reads return.
/// They are dropped and their `Drop` runs, but a panic in it reaches no
/// read: the runtime drops them with the panic swallowed, or in a run it
/// discards.
///
/// Under `Shared`, displaced values are reclaimed through hazard
/// pointers, so a value's `Drop` may run inside a later, unrelated
/// retire on any thread. When such a `Drop` panics, the rest of that
/// reclaim batch is leaked, never dropped and never freed; the runtime
/// stays usable.
// `repr(C)` with `dep_stack` first, because every dep read inside a
// compute passes it to an out-of-line `record_dep`, and at offset 0 that
// pointer is `self`.
#[repr(C)]
pub struct Runtime<C: Cells> {
    pub(crate) dep_stack: <C as Cells>::DepStack,
    pub(crate) nodes: SegmentedNodes<C>,
    /// Read with no lock; written only under `inner.write()`, through
    /// `Inner::arena_tags`.
    pub(crate) arenas: ArenaRegistry<C>,
    pub(crate) inner: OrderedLock<Inner<C>, C>,
    /// Taken only inside an `inner` section or with no lock held.
    pub(crate) watch_queue: OrderedLock<watch_impl::WatchQueue, C>,
    pub(crate) revision: <C as Cells>::U64,
    /// Episodes landed on this runtime, by every thread: an episode can
    /// land a member Clean over nodes it leaves New or Dirty, which a
    /// walk another thread planned earlier would go on to compute.
    pub(crate) episodes_landed: <C as Cells>::U64,
    pub(crate) walk: <C as Cells>::Watermark,
    pub(crate) runtime_id: RuntimeId,
    /// `1` when `get_traced` is actively recording. Checked on every
    /// `compute_one` via a Relaxed load (~1 ns when disarmed) so the
    /// non-tracing hot path pays no measurable cost.
    pub(crate) tracing_armed: <C as Cells>::U8,
    /// The frame budget: a replayable frame's read at this many frames
    /// or more replays instead of nesting.
    pub(crate) max_frames: std::sync::atomic::AtomicU32,
    /// The byte budget, or 0 when it is off.
    pub(crate) max_stack_bytes: std::sync::atomic::AtomicUsize,
    /// Settles, for another runtime's first Clean read from one of this
    /// runtime's frames, whether the frame keeps such reads.
    keep_resolver: cycle::KeepResolver,
    /// `set_max_frames` has been called, so a byte budget set later keeps
    /// its frame budget instead of taking the default's place.
    frames_set: std::sync::atomic::AtomicBool,
    budget_entry: replay_impl::BudgetEntry,
    /// `INCR_TEST_ALL_REPLAYABLE`: every query not marked
    /// `NOT_REPLAYABLE` is replayable.
    pub(crate) all_replayable: bool,
    pub(crate) provenance: provenance_impl::Table<C>,
    tickets: C::Lock<contain_impl::Tickets>,
    tickets_held: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    pub(crate) park_hook: std::sync::OnceLock<ParkHook>,
    #[cfg(test)]
    pub(crate) bag_hook: std::sync::OnceLock<BagHook>,
    #[cfg(test)]
    first_visits: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    expansions: std::sync::atomic::AtomicU64,
    /// Group verifications failed by a member's `verified_at` of 0.
    #[cfg(test)]
    pub(crate) group_zero_fails: std::sync::atomic::AtomicU64,
    /// Visit entries every `ensure_clean` walk on this runtime popped.
    #[cfg(test)]
    pub(crate) walk_pops: std::sync::atomic::AtomicU64,
    /// The node of every group verification, in order.
    #[cfg(test)]
    pub(crate) group_verifies: std::sync::Mutex<Vec<u32>>,
    /// Every doom, in order, as the doom point saw it.
    #[cfg(test)]
    pub(crate) dooms: std::sync::Mutex<Vec<replay_impl::DoomSeen>>,
    /// Walk skips of a claim this thread holds anchored at a virtual
    /// frame's position, whose taint-memo entries the context owns.
    #[cfg(test)]
    pub(crate) virtual_anchor_skips: std::sync::atomic::AtomicU64,
    /// Reads of a held root that went past `cycle_read` to resume a doomed
    /// walk, and walks of a held root resumed.
    #[cfg(test)]
    pub(crate) resumes_past_held: [std::sync::atomic::AtomicU64; 2],
    /// Walk entries revisited because a claim wait ended with them Dirty.
    #[cfg(test)]
    pub(crate) wait_revisits: std::sync::atomic::AtomicU64,
    /// Every episode start, suspension, resumption and member run, in order.
    #[cfg(test)]
    pub(crate) episodes_seen: std::sync::Mutex<Vec<cycle_impl::EpisodeSeen>>,
    /// Every unwind a reading frame took from a pending delivery, in order.
    #[cfg(test)]
    pub(crate) deliveries: std::sync::Mutex<Vec<replay_impl::DeliverySeen>>,
    /// Drop unwinds raised.
    #[cfg(test)]
    pub(crate) drop_unwinds: std::sync::atomic::AtomicU32,
    /// Speculative runs' withheld landings of this revision.
    pub(crate) stashes: OrderedLock<replay_impl::Stashes, C>,
    /// How many stashes `stashes` holds, so a claim can skip its lock.
    pub(crate) stash_count: C::U32,
    /// Speculative chains a driver dropped with no unwind.
    #[cfg(test)]
    pub(crate) cooperative_drops: std::sync::atomic::AtomicU32,
    /// Walk entry claims released unrun after the episode check.
    #[cfg(test)]
    pub(crate) unrun_unclaims: std::sync::atomic::AtomicU32,
    /// Waits on another thread's doom reservation, for tests.
    #[cfg(test)]
    pub(crate) reservation_waits: std::sync::atomic::AtomicU32,
    /// Pops that waited for another thread's claim of a node this thread
    /// reserves to come back, for tests.
    #[cfg(test)]
    pub(crate) give_back_waits: std::sync::atomic::AtomicU32,
    /// Every entry a driver ran, as it found its context.
    #[cfg(test)]
    pub(crate) entries: std::sync::Mutex<Vec<replay_impl::EntrySeen>>,
    /// Episodes started on this runtime.
    #[cfg(test)]
    pub(crate) episodes: std::sync::atomic::AtomicU64,
    /// Episode validations run on this runtime.
    #[cfg(test)]
    pub(crate) validations: std::sync::atomic::AtomicU64,
}

/// The value of [`Runtime::doom_gate`].
#[cfg(all(test, not(loom)))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DoomGate(pub(crate) bool);

/// The default frame budget: 256, or 32 when incr-core is built with
/// debug assertions, whose frames are 14 to 18 times larger. A chain at
/// the budget is deepest once replay runs it, at about 1.5 KiB a nested
/// read (1.86 KiB under `Shared`) in release and 26 KiB in debug: under
/// 480 KiB of a spawned thread's 2 MiB stack in release, leaving each
/// closure on the chain about 6 KiB of its own, and under 840 KiB in
/// debug, leaving about 38 KiB, room for a debug closure's local buffers.
pub const DEFAULT_MAX_FRAMES: u32 = if cfg!(debug_assertions) { 32 } else { 256 };

/// The default frame budget and whether every query is replayable, from
/// `INCR_MAX_DEPTH` and `INCR_TEST_ALL_REPLAYABLE`, read once.
fn replay_defaults() -> (u32, bool) {
    static DEFAULTS: std::sync::OnceLock<(u32, bool)> = std::sync::OnceLock::new();
    *DEFAULTS.get_or_init(|| {
        let frames = std::env::var("INCR_MAX_DEPTH")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MAX_FRAMES);
        let all = std::env::var("INCR_TEST_ALL_REPLAYABLE").is_ok_and(|v| v == "1");
        (frames, all)
    })
}

impl<C: Cells> Default for Runtime<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cells> Runtime<C> {
    pub fn new() -> Self {
        let runtime_id = RuntimeId::allocate();
        Self {
            nodes: SegmentedNodes::new(),
            arenas: ArenaRegistry::new(),
            inner: OrderedLock::new(LockKind::Inner, Inner::new()),
            watch_queue: OrderedLock::new(LockKind::WatchQueue, watch_impl::WatchQueue::new()),
            revision: C::new_u64(2),
            episodes_landed: C::new_u64(0),
            walk: C::new_watermark(2),
            dep_stack: <C::DepStack as DepStack>::new(),
            runtime_id,
            tracing_armed: C::new_u8(0),
            max_frames: std::sync::atomic::AtomicU32::new(replay_defaults().0),
            max_stack_bytes: std::sync::atomic::AtomicUsize::new(0),
            keep_resolver: cycle::KeepResolver {
                keeps: Self::resolve_keep,
            },
            frames_set: std::sync::atomic::AtomicBool::new(false),
            budget_entry: replay_impl::BudgetEntry::new(runtime_id, replay_defaults().0),
            all_replayable: replay_defaults().1,
            provenance: provenance_impl::table::<C>(),
            tickets: contain_impl::tickets::<C>(),
            tickets_held: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            park_hook: std::sync::OnceLock::new(),
            #[cfg(test)]
            bag_hook: std::sync::OnceLock::new(),
            #[cfg(test)]
            first_visits: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            expansions: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            group_zero_fails: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            walk_pops: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            group_verifies: std::sync::Mutex::new(Vec::new()),
            #[cfg(test)]
            dooms: std::sync::Mutex::new(Vec::new()),
            #[cfg(test)]
            virtual_anchor_skips: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            resumes_past_held: Default::default(),
            #[cfg(test)]
            wait_revisits: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            episodes_seen: std::sync::Mutex::new(Vec::new()),
            #[cfg(test)]
            deliveries: std::sync::Mutex::new(Vec::new()),
            #[cfg(test)]
            drop_unwinds: std::sync::atomic::AtomicU32::new(0),
            stashes: OrderedLock::new(LockKind::Stashes, replay_impl::Stashes::default()),
            stash_count: C::new_u32(0),
            #[cfg(test)]
            cooperative_drops: std::sync::atomic::AtomicU32::new(0),
            #[cfg(test)]
            unrun_unclaims: std::sync::atomic::AtomicU32::new(0),
            #[cfg(test)]
            reservation_waits: std::sync::atomic::AtomicU32::new(0),
            #[cfg(test)]
            give_back_waits: std::sync::atomic::AtomicU32::new(0),
            #[cfg(test)]
            entries: std::sync::Mutex::new(Vec::new()),
            #[cfg(test)]
            episodes: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            validations: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// First visits made by every `ensure_clean` walk on this runtime so far.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn first_visits(&self) -> u64 {
        self.first_visits.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Dep expansions made by every `ensure_clean` walk on this runtime so far.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn expansions(&self) -> u64 {
        self.expansions.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    fn park(&self, point: ParkPoint) {
        if let Some(hook) = self.park_hook.get() {
            hook(point);
        }
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn park(&self, _point: ParkPoint) {}

    /// [`Self::park`] at a point the attempt's bag is carried past.
    #[cfg(test)]
    fn park_with_bag(&self, point: ParkPoint, bag: &mut Deferred<C>) {
        self.park(point);
        if let Some(item) = self.bag_hook.get().and_then(|hook| hook(point)) {
            bag.push_erased(item);
        }
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn park_with_bag(&self, _point: ParkPoint, _bag: &mut Deferred<C>) {}

    #[inline(always)]
    fn tracing_is_armed(&self) -> bool {
        C::u8_load_relaxed(&self.tracing_armed) == 1
    }

    #[inline(always)]
    fn record_trace(&self, id: NodeId, action: crate::trace::TraceAction) {
        if self.tracing_is_armed() {
            self.push_trace(id, action);
        }
    }

    #[cold]
    #[inline(never)]
    fn push_trace(&self, id: NodeId, action: crate::trace::TraceAction) {
        self.inner
            .write()
            .trace_log
            .push(crate::trace::NodeTrace { id, action });
    }

    #[inline]
    fn current_revision(&self) -> u64 {
        C::u64_load_acquire(&self.revision)
    }

    fn bump_revision(&self) -> u64 {
        // fetch_add, not load+store: two concurrent set() calls must not
        // collapse into one revision or the cutoff protocol conflates
        // their changes.
        C::u64_fetch_add(&self.revision, 2) + 2
    }

    /// Create an input node with an initial value.
    pub fn create_input<T: Value>(&self, value: T) -> Incr<T> {
        assert!(
            !self.dep_stack.current_frame_active(self.runtime_id),
            "create_input called during compute; not permitted (a compute creates its \
             nodes through a NodeTable)",
        );
        self.create_input_unchecked(value)
    }

    /// Internal: create an input without the dep-stack-empty check.
    /// Used by operators like `group_by` that need to allocate
    /// sub-collection version nodes lazily from inside a compute closure.
    /// The caller is responsible for ensuring the new node is not a dep
    /// of the currently-computing node (i.e., the new node is downstream
    /// of the operator, not upstream).
    #[inline(always)]
    pub(crate) fn create_input_unchecked<T: Value>(&self, value: T) -> Incr<T> {
        value_code::check();
        let revision = self.current_revision();
        let (slot, generation) = {
            let mut inner = self.inner.write();
            let (type_tag, arena) = self.arenas.ensure_tagged::<T>(&mut inner.arena_tags);
            let arena_slot = arena.reserve_with(value);
            let slot = if let Some(slot) = inner.free_nodes.pop() {
                self.nodes
                    .get(slot)
                    .reinit_recycled(type_tag, arena_slot, revision, true);
                slot
            } else {
                let node = NodeData::<C>::new_input(type_tag, arena_slot, revision);
                let slot = self.nodes.push(node);
                inner.compute_fns.push(None);
                inner.dependents.push(DependentList::new());
                slot
            };
            (slot, self.nodes.get(slot).generation())
        };
        Incr::new(slot, generation, self.runtime_id)
    }

    /// Create a compute (query) node. Dependencies are tracked
    /// automatically: every `rt.get(other)` call inside `f` records
    /// `other` as a dep.
    pub fn create_query<T, F>(&self, f: F) -> Incr<T>
    where
        T: Value,
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.create_query_erased::<true, _, _>(f, ErasedOpts::NONE)
    }

    /// [`Self::create_query`] with a [`CycleStrategy::Fallback`]: in a
    /// dependency cycle the query's value is `g` of the cycle.
    pub fn create_query_with_fallback<T, F, G>(&self, f: F, g: G) -> Incr<T>
    where
        T: Value,
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
        G: Fn(&Cycle) -> T + Send + Sync + 'static,
    {
        self.create_query_with(
            f,
            QueryOpts {
                cycle: CycleStrategy::fallback(g),
                ..Default::default()
            },
        )
    }

    /// [`Self::create_query`] with options: the query's cycle strategy.
    pub fn create_query_with<T, F>(&self, f: F, opts: QueryOpts<T>) -> Incr<T>
    where
        T: Value,
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.create_query_erased::<true, _, _>(f, cycle::erase_opts(opts))
    }

    /// Creates a collection operator's query, `NOT_REPLAYABLE` whatever
    /// `INCR_TEST_ALL_REPLAYABLE` says: its run consumes its upstream's
    /// log through a cursor, so a doomed run could not be re-run from
    /// the start.
    pub(crate) fn create_operator_query<T, F>(&self, f: F) -> Incr<T>
    where
        T: Value,
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.create_query_erased::<true, _, _>(
            f,
            ErasedOpts {
                strategy: None,
                flags: NOT_REPLAYABLE,
            },
        )
    }

    /// Takes the strategy erased so that [`Self::create_query`] builds and
    /// drops no `CycleStrategy` value. `CHECKED` is false for a table miss,
    /// which may create inside a compute; a const parameter keeps the
    /// checked instance's code what it was before tables existed.
    pub(crate) fn create_query_erased<const CHECKED: bool, T, F>(
        &self,
        f: F,
        opts: ErasedOpts,
    ) -> Incr<T>
    where
        T: Value,
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        if CHECKED {
            assert!(
                !self.dep_stack.current_frame_active(self.runtime_id),
                "create_query called during compute; not permitted (a compute creates its \
                 nodes through a NodeTable)",
            );
        }
        value_code::check();

        let (slot, generation) = {
            let mut inner = self.inner.write();
            let (type_tag, arena) = self.arenas.ensure_tagged_shared::<T>(&mut inner.arena_tags);
            let arena_slot = arena.reserve();
            let slot = if let Some(slot) = inner.free_nodes.pop() {
                self.nodes
                    .get(slot)
                    .reinit_recycled(type_tag, arena_slot, 0, false);
                slot
            } else {
                let node = NodeData::<C>::new_query(type_tag, arena_slot);
                let slot = self.nodes.push(node);
                inner.compute_fns.push(None);
                inner.dependents.push(DependentList::new());
                slot
            };

            // A first compute writes unconditionally; a recompute
            // compares and publishes in one slot session, and an equal
            // value is the early cutoff. A frame that read an open cycle,
            // or an episode's member run, keeps the value in its control
            // block instead.
            let arena_inner = arena;
            let compute: ComputeFn<C> = Box::new(
                move |rt: &Runtime<C>, slot: u32, ctl: &ComputeCtl| -> Outcome {
                    let new_value = f(rt);
                    let node = rt.nodes.get(slot);
                    let outcome = match ctl.flags.get()
                        & (cycle::RECOMPUTE
                            | cycle::BUFFERED
                            | cycle::LOW
                            | cycle::DOOMED
                            | cycle::SPECULATIVE)
                    {
                        0 => {
                            ctl.wrote.set(1);
                            drop(arena_inner.write(node.arena_slot(), new_value));
                            Outcome::Changed
                        }
                        cycle::RECOMPUTE => {
                            if arena_inner.write_if_changed(
                                node.arena_slot(),
                                new_value,
                                &ctl.wrote,
                            ) {
                                Outcome::Changed
                            } else {
                                Outcome::Unchanged
                            }
                        }
                        _ => return ctl.buffer(new_value),
                    };
                    reclaim_probe::reached(RetireSite::ComputeWrite);
                    outcome
                },
            );

            inner.compute_fns[slot as usize] = Some(compute);
            if let Some(strategy) = opts.strategy {
                inner.cycle_strategies.insert(slot, strategy);
                let flags = self.nodes.get_with_flags(slot).1;
                C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) | HAS_STRATEGY);
            }
            if opts.flags != 0 {
                let flags = self.nodes.get_with_flags(slot).1;
                C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) | opts.flags);
            }
            (slot, self.nodes.get(slot).generation())
        };
        Incr::new(slot, generation, self.runtime_id)
    }

    /// Reject handles minted by a different runtime and handles whose
    /// node was deleted (the slot's generation moved past the handle's).
    /// Real branches, not debug_asserts: in release a foreign or stale
    /// handle would silently read or write another node's slot in the
    /// same type arena, which is silent data corruption. Cost is two
    /// always-predicted compares on the hot path.
    #[inline(always)]
    fn check_handle<T: Value>(&self, handle: Incr<T>) {
        self.check_runtime(handle);
        let node = self.nodes.get(handle.slot());
        if node.generation() != handle.generation() {
            self.stale_handle(handle);
        }
    }

    #[inline(always)]
    fn check_runtime<T: Value>(&self, handle: Incr<T>) {
        if handle.runtime_id() != self.runtime_id {
            self.foreign_handle(handle);
        }
    }

    #[cold]
    #[inline(never)]
    fn foreign_handle<T: Value>(&self, handle: Incr<T>) -> ! {
        if handle.runtime_id() == RuntimeId::PENDING {
            if self.pending_in_reach(handle.erase()) {
                panic!(
                    "incr-core: a pending Incr<{}> handle can only be read",
                    std::any::type_name::<T>(),
                );
            }
            self.pending_out_of_reach();
        }
        panic!(
            "incr-core: Incr<{}> handle from a foreign runtime (handle was minted by \
             runtime {}, used on runtime {})",
            std::any::type_name::<T>(),
            handle.runtime_id().get(),
            self.runtime_id.get(),
        );
    }

    /// `get` and `try_get` of a handle another runtime minted, or of a
    /// pending one: a pending handle of this runtime read in a doomed
    /// frame reads its type's placeholder and records no dependency;
    /// any other panics.
    #[cold]
    #[inline(never)]
    fn read_mismatch<T: Value>(&self, handle: Incr<T>) -> T {
        if handle.runtime_id() == RuntimeId::PENDING && self.pending_in_reach(handle.erase()) {
            return self.placeholder::<T>().expect(
                "incr-core: a pending handle is minted only for a type with a placeholder",
            );
        }
        self.foreign_handle(handle)
    }

    /// Whether a pending `handle` was minted by this runtime and this
    /// thread is in a doomed frame of it, the only place it means anything.
    fn pending_in_reach(&self, handle: ErasedIncr) -> bool {
        let minted_by = u64::from(handle.slot()) | (u64::from(handle.generation()) << 32);
        minted_by == self.runtime_id.get() && self.frame_doomed()
    }

    #[cold]
    fn pending_out_of_reach(&self) -> ! {
        panic!("incr-core: pending handle used outside the replay attempt that produced it");
    }

    /// Whether this thread is inside a compute of any runtime. Every
    /// runtime on a thread shares its frame stack, and a compute of
    /// another runtime holds that runtime's claim as surely as one of this
    /// runtime's does.
    #[doc(hidden)]
    pub fn frame_active(&self) -> bool {
        self.dep_stack.any_compute_frame()
    }

    /// Whether this runtime's active frame on this thread is doomed.
    pub(crate) fn frame_doomed(&self) -> bool {
        self.dep_stack
            .active_frame(self.runtime_id)
            .is_some_and(|(ctl, _)| {
                // SAFETY: an active frame's control block is live while the
                // frame is on the stack (see `Frame::ctl`).
                unsafe { (*ctl).has(cycle::DOOMED) }
            })
    }

    /// A node that dooms a replayable frame reading it at budget 1:
    /// reading it gives `DoomGate(false)`, its placeholder, in the doomed
    /// run and `DoomGate(true)` in the retry the driver makes once the
    /// gate has landed. Sets the frame budget to 1.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn doom_gate(&self) -> Incr<DoomGate> {
        self.set_max_frames(1);
        self.set_placeholder(DoomGate(false));
        self.create_query(|_| DoomGate(true))
    }

    /// The handle a table miss in a doomed frame returns: no node, only
    /// this runtime's id, so it reads `T`'s placeholder inside the frame
    /// and panics anywhere else. Panics if `T` has no placeholder.
    pub(crate) fn pending_handle<T: Value>(&self) -> Incr<T> {
        if self.placeholder::<T>().is_none() {
            panic!(
                "incr-core: a table miss in a doomed frame needs a placeholder for {}; \
                 register one with set_placeholder",
                std::any::type_name::<T>(),
            );
        }
        let id = self.runtime_id.get();
        Incr::new(id as u32, (id >> 32) as u32, RuntimeId::PENDING)
    }

    /// Sets the value a pending handle of type `T` reads. Panics if `T`
    /// already has one, so readers never race a replacement.
    pub fn set_placeholder<T: Value>(&self, value: T) {
        value_code::check();
        let set = {
            let mut inner = self.inner.write();
            let (_, arena) = self.arenas.ensure_tagged::<T>(&mut inner.arena_tags);
            arena.set_placeholder(value)
        };
        assert!(
            set,
            "incr-core: placeholder already set for {}",
            std::any::type_name::<T>(),
        );
    }

    /// Sets the frame budget, [`DEFAULT_MAX_FRAMES`] unless `INCR_MAX_DEPTH`
    /// says otherwise: a read made by a replayable query's compute when
    /// this runtime's frames on the thread number `n` or more (every
    /// runtime's, under `Shared`) replays instead of nesting, if the read's
    /// type has a placeholder. Replay changes where computes run, never a
    /// result. A compute running on a thread when that thread first meets
    /// a budget change is never replayed for it, since its reads before
    /// were made under the budget before.
    ///
    /// Panics if `n` is 0.
    pub fn set_max_frames(&self, n: u32) {
        assert!(n > 0, "incr-core: the frame budget is at least 1");
        self.set_budget(|| {
            self.frames_set
                .store(true, std::sync::atomic::Ordering::Relaxed);
            (
                n,
                self.max_stack_bytes
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
        });
    }

    /// Sets a byte budget beside the frame budget, off when `n` is 0 (the
    /// default): a replayable query's read also replays when the stack
    /// this thread has used since its outermost engine call exceeds `n`
    /// bytes. Frames differ in size with the closure and the build, so
    /// the frame count alone does not bound the stack. Like
    /// [`Self::set_max_frames`], it wins over `INCR_MAX_DEPTH`: a runtime
    /// that has not set its frame budget gets [`DEFAULT_MAX_FRAMES`]. A
    /// change spares running computes as [`Self::set_max_frames`]'s does.
    pub fn set_max_stack_bytes(&self, n: usize) {
        self.set_budget(|| {
            let frames = if self.frames_set.load(std::sync::atomic::Ordering::Relaxed) {
                self.max_frames.load(std::sync::atomic::Ordering::Relaxed)
            } else {
                DEFAULT_MAX_FRAMES
            };
            (frames, n)
        });
    }

    /// A clone of `T`'s placeholder, if it has one.
    pub(crate) fn placeholder<T: Value>(&self) -> Option<T> {
        let tag = self.inner.read().arena_tags.tag::<T>()?;
        let arena = self.arenas.tagged_arena::<T>(tag)?;
        value_code::run(|| arena.placeholder().cloned())
    }

    /// A [`NodeTable`] bound to this runtime, hashing with
    /// [`FxBuildHasher`].
    pub fn create_table<K: TableKey, T: Value>(&self) -> NodeTable<K, T, C> {
        NodeTable::new(self.runtime_id, FxBuildHasher)
    }

    /// [`Self::create_table`] with another hasher, such as one resistant
    /// to chosen collisions for keys from untrusted input.
    pub fn create_table_with_hasher<K: TableKey, T: Value, S: std::hash::BuildHasher>(
        &self,
        hasher: S,
    ) -> NodeTable<K, T, C, S> {
        NodeTable::new(self.runtime_id, hasher)
    }

    #[cold]
    #[inline(never)]
    fn stale_handle<T: Value>(&self, handle: Incr<T>) -> ! {
        panic!(
            "incr-core: stale Incr<{}> handle: the node at slot {} was deleted (handle \
             generation {}, slot generation {})",
            std::any::type_name::<T>(),
            handle.slot(),
            handle.generation(),
            self.nodes.get(handle.slot()).generation(),
        );
    }

    /// A read that loaded the node Clean found its slot empty: under
    /// `Shared` another thread's set and episode can land it as a cycle
    /// member, which clears the slot, between the load and the read. A
    /// Clean reload is read again, so only a Clean node with an empty slot
    /// is the invariant panic; `Err` hands back any other state, which the
    /// read goes on from as if it had loaded it first. A diverted read's
    /// `type_tag` is `DIVERTED_TAG`, and [`Self::diverted_clean_read`]
    /// answers it, with `Err(Clean)` for a read that goes on as a slow one.
    #[cold]
    #[inline(never)]
    fn reload_after_empty_slot<T: Value>(
        &self,
        handle: Incr<T>,
        node: &NodeData<C>,
        type_tag: u16,
        arena_slot: u32,
    ) -> Result<T, NodeState> {
        if type_tag == crate::dep_stack::DIVERTED_TAG {
            return self.diverted_clean_read(handle, node);
        }
        if C::SINGLE_THREADED {
            self.read_failed(handle);
        }
        match state::load::<C>(node.state_cell()) {
            NodeState::Clean => Ok(self.read_slot(handle, node, type_tag, arena_slot)),
            now => Err(now),
        }
    }

    /// A Clean read the top frame diverted (see `Frame::read_rt`). The
    /// frame's own read that a replay context holds a delivery for goes on
    /// as the slow read the unbounded run made before the node landed
    /// Clean, which takes the delivery; any other is the fast read.
    #[cold]
    #[inline(never)]
    fn diverted_clean_read<T: Value>(
        &self,
        handle: Incr<T>,
        node: &NodeData<C>,
    ) -> Result<T, NodeState> {
        let rt = self.runtime_id;
        let (dep, generation) = (NodeId(handle.slot()), handle.generation());
        if let Some(f) = self.dep_stack.diverted_read(rt) {
            let word = value_code::thread_word();
            if self.delivery_waits(f, handle, word) {
                return Err(NodeState::Clean);
            }
            self.dep_stack.record_diverted_read(rt, dep, generation);
            if word & value_code::REPLAY_BIT == 0 {
                self.dep_stack.end_diverted_reads(rt);
            } else if word & value_code::COUNTER_MASK == 0 && !replay::held() {
                // A `Drop` that an unwind out of a `replay::with` closure runs
                // is inside the contexts' borrow, so only other reads look
                // there.
                self.undivert_when_taken(f);
            }
        } else if let Some(reader) = self.dep_stack.foreign_keeper(rt) {
            let word = value_code::thread_word();
            if word & value_code::COUNTER_MASK == 0 && !replay::held() {
                if word & value_code::REPLAY_BIT == 0 {
                    return self.foreign_clean_read(handle, node, reader);
                }
                // A retry's read may take what its doomed run kept, which
                // the slow path's foreign read does.
                return Err(NodeState::Clean);
            }
            // Not kept, so the frame must not be doomed: a retry would
            // read this runtime again.
            self.dep_stack.hold_foreign_reader();
        }
        let (type_tag, arena_slot) = self.fields_under(handle, node);
        match self.try_read_slot(handle, node, type_tag, arena_slot) {
            Some(value) => Ok(value),
            None => self.reload_after_empty_slot(handle, node, type_tag, arena_slot),
        }
    }

    /// The read of a Clean node found no value behind its tag and arena
    /// slot, or found one under a moved generation. A moved generation
    /// is a stale handle; an unmoved one means the fields and the arena
    /// disagree, which no delete can cause.
    #[cold]
    #[inline(never)]
    fn read_failed<T: Value>(&self, handle: Incr<T>) -> ! {
        if self.nodes.get(handle.slot()).generation() != handle.generation() {
            self.stale_handle(handle);
        }
        panic!(
            "incr-core: internal invariant violated: node {} is Clean but its Incr<{}> \
             value is missing from the arena",
            handle.slot(),
            std::any::type_name::<T>(),
        );
    }

    /// Delete a node and recycle its slot. The node must have no
    /// dependents (delete downstream nodes first) and must not be
    /// deleted from inside its own compute. Any in-flight compute on the
    /// node is waited out, exactly like a reader. After deletion every
    /// surviving copy of the handle panics with a stale-handle message
    /// on use; the slot and its arena storage are reused by future
    /// create calls under a bumped generation.
    ///
    /// Panics where [`Runtime::try_delete_node`] returns an error, and on
    /// a handle that is stale or from another runtime.
    pub fn delete_node<T: Value>(&self, handle: Incr<T>) {
        self.check_handle(handle);
        let root = handle.erase();
        match self.enter(|| self.delete_strict(root)) {
            Ok(()) => {}
            // check_handle passed, so a concurrent delete got there first.
            Err(StrictRefusal::Gone) => {
                panic!("incr-core: node {} deleted twice concurrently", root.slot())
            }
            Err(StrictRefusal::HasDependents(slot)) if slot == root.slot() => {
                Self::panic_has_dependents(root.slot())
            }
            Err(StrictRefusal::HasDependents(child)) => panic!(
                "incr-core: delete_node on node {} whose owned node {} still has \
                 dependents; delete downstream nodes first",
                root.slot(),
                child,
            ),
        }
    }

    /// Delete a node and every node it owns, like
    /// [`Runtime::delete_node`], or report why not and change nothing.
    ///
    /// The dependents checks and the claims of all those nodes happen
    /// under one lock hold, so under `Shared` a query that computes
    /// against the node concurrently either lands before the check, and
    /// this returns [`DeleteError::HasDependents`] with the node intact,
    /// or after the claim, and finds the node deleted. A query that was
    /// created but never ran has no edge yet and does not count.
    ///
    /// Panics if called from inside the node's own compute.
    pub fn try_delete_node(&self, handle: impl Into<ErasedIncr>) -> Result<(), DeleteError> {
        let root = handle.into();
        if root.runtime_id() != self.runtime_id {
            return Err(DeleteError::NotAlive);
        }
        self.enter(|| self.delete_strict(root))
            .map_err(|refusal| match refusal {
                StrictRefusal::Gone => DeleteError::NotAlive,
                StrictRefusal::HasDependents(_) => DeleteError::HasDependents,
            })
    }

    fn panic_has_dependents(slot: u32) -> ! {
        panic!(
            "incr-core: delete_node on node {} which still has dependents; \
             delete downstream nodes first",
            slot,
        )
    }

    /// A node with a virtual frame on this thread stands for a frame of
    /// its own below the caller, though nothing holds its claim.
    #[inline]
    fn assert_not_own_compute(&self, id: NodeId, generation: u32) {
        assert!(
            !self.holds_claim(id)
                && (value_code::thread_word() & value_code::REPLAY_BIT == 0
                    || self.no_virtual_frame(id.0, generation)),
            "incr-core: delete_node called from inside node {}'s own compute",
            id.0,
        );
    }

    /// Out of line so the scan does not stop every delete inlining the
    /// assert above.
    #[cold]
    #[inline(never)]
    fn no_virtual_frame(&self, slot: u32, generation: u32) -> bool {
        self.virtual_position(slot, generation).is_none()
    }

    /// The strict delete behind `delete_node` and `try_delete_node`:
    /// `root` and everything it owns, transitively, or nothing.
    ///
    /// A root that owns nothing takes the single-node claim, `Dead` under
    /// the read hold that checks its dependents, and a set holding it is
    /// waited out before it is retired. An owner's whole set goes through
    /// [`Runtime::claim_set`] instead.
    fn delete_strict(&self, root: ErasedIncr) -> Result<(), StrictRefusal> {
        debug_assert_eq!(root.runtime_id(), self.runtime_id);
        value_code::check();
        let id = NodeId(root.slot());
        let node = self.nodes.get(root.slot());
        if node.generation() != root.generation() {
            return Err(StrictRefusal::Gone);
        }
        self.assert_not_own_compute(id, root.generation());
        self.refuse_delete_under_own_set(id);
        self.park(ParkPoint::DeleteAfterHandleCheck(id));

        let mut spins: u32 = 0;
        match self.claim_single(root, &mut spins) {
            SingleClaim::Claimed => {}
            SingleClaim::Gone => return Err(StrictRefusal::Gone),
            SingleClaim::HasDependents => return Err(StrictRefusal::HasDependents(root.slot())),
            SingleClaim::Owner => return self.delete_owned_set(root, &mut spins),
        }
        self.park(ParkPoint::DeleteAfterCheck(id));
        self.wait_out_set(root, &mut spins);
        self.release_claimed(root).drain(self);
        Ok(())
    }

    /// Claims `handle`'s node `Dead` for a delete, waiting out a compute
    /// on it. The check and the claim share one read hold so no publish
    /// (which pushes reverse edges under the write lock) can land between
    /// them; clearing dependents at the release would otherwise drop its
    /// edge silently. The generation is rechecked under the hold because
    /// a delete and recycle between the load of the state and the CAS
    /// would otherwise let the CAS claim the slot's next occupant.
    #[inline(always)]
    fn claim_single(&self, handle: ErasedIncr, spins: &mut u32) -> SingleClaim {
        let id = NodeId(handle.slot());
        let node = self.nodes.get(handle.slot());
        loop {
            let cur = state::load::<C>(node.state_cell());
            if node.generation() != handle.generation() {
                return SingleClaim::Gone;
            }
            match cur {
                NodeState::Computing | NodeState::ComputingDirty => {
                    self.refuse_wait_on_computing(id);
                    self.spin_on(id, spins, || {
                        !is_computing(state::load::<C>(node.state_cell()))
                    });
                }
                NodeState::Dead => {
                    // Another delete claimed the node and will bump the
                    // generation under its next write lock. Returning now
                    // would let a cascade re-find this slot in its
                    // parent's dependents and spin on it instead.
                    self.wait_out_claim(handle, spins);
                    return SingleClaim::Gone;
                }
                NodeState::Deleting => {
                    *spins += 1;
                    self.wait_out_deleting(id, *spins);
                }
                _ => {
                    let inner = self.inner.read();
                    if node.generation() != handle.generation() {
                        return SingleClaim::Gone;
                    }
                    // An owner's dependents may sit inside its owned set,
                    // so only the set's claim can judge them.
                    if self.owns_nodes(&inner, handle.slot()) {
                        return SingleClaim::Owner;
                    }
                    if !inner.dependents[handle.slot() as usize].is_empty() {
                        return SingleClaim::HasDependents;
                    }
                    if state::try_transition::<C>(node.state_cell(), cur, NodeState::Dead).is_ok() {
                        return SingleClaim::Claimed;
                    }
                }
            }
        }
    }

    /// One backoff on a node found `Deleting`. Only another thread's
    /// `claim_set` holds that state, inside a write hold that runs no
    /// user code and waits on nothing, so the wait registers nowhere.
    #[cold]
    #[inline(never)]
    pub(crate) fn wait_out_deleting(&self, id: NodeId, spins: u32) {
        if C::SINGLE_THREADED {
            panic!(
                "incr-core: internal invariant violated: node {} is Deleting outside the \
                 delete that holds it",
                id.0,
            );
        }
        C::backoff(spins);
    }

    /// Waits, with no guard held, until the delete that claimed `handle`'s
    /// node `Dead` has bumped its generation under the write lock.
    fn wait_out_claim(&self, handle: ErasedIncr, spins: &mut u32) {
        self.park(ParkPoint::DeleteWaitsOnClaim(NodeId(handle.slot())));
        let node = self.nodes.get(handle.slot());
        while node.generation() == handle.generation() {
            if C::SINGLE_THREADED {
                panic!(
                    "incr-core: node {} is claimed by a delete on this thread that has not \
                     unhooked it yet; a Drop or callback must not delete a node whose own \
                     delete is in progress",
                    handle.slot(),
                );
            }
            self.spin_on(NodeId(handle.slot()), spins, || {
                node.generation() != handle.generation()
            });
        }
    }

    /// Marks a node this thread claimed `Dead` as deleted, so no set
    /// claims it any more, and waits out a set that claimed it first,
    /// before the delete retires it. Waits with no guard held; the set
    /// takes none before its release.
    fn wait_out_set(&self, handle: ErasedIncr, spins: &mut u32) {
        let node = self.nodes.get(handle.slot());
        if !node.mark_deleted() {
            return;
        }
        loop {
            self.refuse_wait_on_own_set(NodeId(handle.slot()));
            self.spin_on(NodeId(handle.slot()), spins, || !node.set_claimed());
            if !node.set_claimed() {
                return;
            }
        }
    }

    /// The handle of the node currently at `slot`. Only meaningful under
    /// an `inner` hold, where the dependents and owned invariants make
    /// the slot's generation the linked node's.
    fn erased_at(&self, slot: u32) -> ErasedIncr {
        ErasedIncr::new(slot, self.nodes.get(slot).generation(), self.runtime_id)
    }

    /// Record `child` as owned by `owner`: it is deleted with the owner,
    /// by strict delete and by the cascade. Both must be live, and
    /// `child` must not own `owner`, transitively, which would make the
    /// owned sets a cycle.
    pub(crate) fn adopt(&self, owner: ErasedIncr, child: ErasedIncr) {
        value_code::check();
        let mut inner = self.inner.write();
        for h in [owner, child] {
            assert!(
                self.is_alive(h),
                "incr-core: adopt of deleted node {}",
                h.slot()
            );
        }
        let mut up = Some(NodeId(owner.slot()));
        while let Some(o) = up {
            assert!(
                o.0 != child.slot(),
                "incr-core: adopt of node {} by node {} would make it own itself",
                child.slot(),
                owner.slot(),
            );
            up = inner.owner_of.get(&o.0).copied();
        }
        self.adopt_locked(&mut inner, owner.slot(), child.slot());
    }

    /// [`Self::adopt`] past its checks, under the caller's write hold.
    pub(crate) fn adopt_locked(&self, inner: &mut Inner<C>, owner: u32, child: u32) {
        let previous = inner.owner_of.insert(child, NodeId(owner));
        assert!(previous.is_none(), "incr-core: node {child} adopted twice");
        inner.owned.entry(owner).or_default().push(NodeId(child));
        let flags = self.nodes.get_with_flags(owner).1;
        C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) | HAS_CHILDREN);
        let flags = self.nodes.get_with_flags(child).1;
        C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) | OWNED);
    }

    /// Whether `slot` owns nodes. The caller holds `inner`, under whose
    /// write hold the flag is kept in step with `Inner::owned`. An empty
    /// map is tested first because it answers for every node with one
    /// load, and the flag spares the hash once any node owns something.
    #[inline]
    fn owns_nodes(&self, inner: &Inner<C>, slot: u32) -> bool {
        !inner.owned.is_empty()
            && C::u8_load_relaxed(self.nodes.get_with_flags(slot).1) & HAS_CHILDREN != 0
    }

    /// The nodes `slot` owns, looking the map up only for an owner.
    #[inline]
    fn owned_by<'a>(&self, inner: &'a Inner<C>, slot: u32) -> Option<&'a Vec<NodeId>> {
        if self.owns_nodes(inner, slot) {
            inner.owned.get(&slot)
        } else {
            None
        }
    }

    /// True iff `handle` names a node of this runtime that has not been
    /// deleted. A handle from another runtime is simply not alive here.
    /// The answer is a snapshot: a concurrent delete can end the node
    /// right after it returns.
    pub fn is_alive(&self, handle: impl Into<ErasedIncr>) -> bool {
        let handle = handle.into();
        if handle.runtime_id() != self.runtime_id {
            return false;
        }
        let node = self.nodes.get(handle.slot());
        // State before generation: a recycle bumps the generation before
        // it Release-stores the new occupant's state, so if the Acquire
        // load below sees that occupant, the generation load sees the bump.
        let state = node.state();
        state != NodeState::Dead && node.generation() == handle.generation()
    }

    /// The deletion path behind every public delete. Never panics on a
    /// dependent or on a node that is already gone; it reports them. An
    /// owned node counts as a dependent: callers delete owned nodes first.
    ///
    /// Invariant: a slot listed in any `dependents` or `owned` list
    /// carries the generation of the node linked there. That is why the
    /// generation is bumped under `inner.write()`, together with the
    /// unhooking, and not at the claim.
    pub(crate) fn delete_erased(&self, handle: ErasedIncr) -> DeleteOutcome {
        assert_eq!(
            handle.runtime_id(),
            self.runtime_id,
            "incr-core: delete of a handle from a foreign runtime",
        );
        value_code::check();
        let slot = handle.slot();
        let id = NodeId(slot);
        let node = self.nodes.get(slot);
        if node.generation() != handle.generation() {
            return DeleteOutcome::AlreadyGone;
        }
        self.assert_not_own_compute(id, handle.generation());
        self.refuse_delete_under_own_set(id);

        let mut spins: u32 = 0;
        match self.claim_single(handle, &mut spins) {
            SingleClaim::Claimed => self.park(ParkPoint::DeleteAfterCheck(id)),
            SingleClaim::Gone => return DeleteOutcome::AlreadyGone,
            SingleClaim::HasDependents | SingleClaim::Owner => return DeleteOutcome::HasDependents,
        }
        self.wait_out_set(handle, &mut spins);
        self.release_claimed(handle).drain(self);
        DeleteOutcome::Deleted
    }

    /// Unhooks and recycles a node this thread claimed `Dead`. The node
    /// must own nothing by now: its owned nodes are released first. Its
    /// closure, observers, value and displaced dep list go into the
    /// returned bag, which the delete drains once it holds no lock.
    #[inline(always)]
    fn release_claimed(&self, handle: ErasedIncr) -> DeleteBag<'_, C> {
        let retired = self.retire_claimed(handle);
        let (drops, dep_list) = {
            let mut inner = self.inner.write();
            self.unhook_retired(&mut inner, retired)
        };
        DeleteBag::new(drops, dep_list)
    }

    /// The part of a release that runs under the delete's claim and
    /// before the hold: the deps and the Dead state are cleared before
    /// the slot can reach the free list, and the displaced list is
    /// retired only at the bag's drain.
    #[inline(always)]
    fn retire_claimed(&self, handle: ErasedIncr) -> Retired<C> {
        let node = self.nodes.get(handle.slot());
        let mut old_deps: Vec<NodeId> = Vec::new();
        node.for_each_dep(|d| old_deps.push(d));
        Retired {
            handle,
            old_deps,
            arena_slot: node.arena_slot(),
            type_tag: node.type_tag(),
            dep_list: node.retire_for_delete(),
        }
    }

    /// The part of a release that runs under `inner.write()`: unhook the
    /// node's edges and ownership, bump its generation and take what the
    /// bag drops later. The closures, observers and value are dropped
    /// only at the bag's drain, after the lock is released, because their
    /// Drop impls are user code that may call back into the runtime.
    #[inline(always)]
    fn unhook_retired(
        &self,
        inner: &mut Inner<C>,
        retired: Retired<C>,
    ) -> (NodeDrops<'_, C>, Option<RetiredDeps<C>>) {
        let slot = retired.handle.slot();
        let id = NodeId(slot);
        let (node, flags) = self.nodes.get_with_flags(slot);
        debug_assert!(
            !inner.owned.contains_key(&slot),
            "incr-core: released node {} still owns nodes",
            slot,
        );
        for d in &retired.old_deps {
            inner.dependents[d.0 as usize].remove(id);
        }
        inner.dependents[slot as usize].clear();
        let bits = C::u8_load_relaxed(flags);
        if let Some(owner) = (bits & OWNED != 0)
            .then(|| inner.owner_of.remove(&slot))
            .flatten()
        {
            let siblings = inner
                .owned
                .get_mut(&owner.0)
                .expect("incr-core: owner_of names an owner with no owned list");
            siblings.retain(|&c| c != id);
            if siblings.is_empty() {
                inner.owned.remove(&owner.0);
                let flags = self.nodes.get_with_flags(owner.0).1;
                C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) & !HAS_CHILDREN);
            }
        }
        let strategy = if bits & HAS_STRATEGY != 0 {
            inner.cycle_strategies.remove(&slot)
        } else {
            None
        };
        if bits & IN_GROUP != 0 {
            inner.member_recs.remove(&slot);
        }
        let children = if bits & HAS_INDEX != 0 {
            inner.children.remove(&slot)
        } else {
            None
        };
        self.unwatch_deleted(slot, bits);
        C::u8_store_relaxed(flags, 0);
        self.bury_tickets(slot, node.generation());
        node.bump_generation();
        let compute = inner.compute_fns[slot as usize].take();
        let observers = inner.observers.take_slot(slot);
        if !inner.labels.is_empty() {
            inner.labels.remove(&slot);
        }
        let arena = self.arenas.erased(retired.type_tag);
        inner.free_nodes.push(slot);
        let drops = NodeDrops {
            children,
            compute,
            strategy,
            observers,
            arena,
            arena_slot: retired.arena_slot,
        };
        (drops, retired.dep_list)
    }

    /// Read the current value of a node. Triggers recomputation of the
    /// minimum necessary subgraph if anything is dirty.
    ///
    /// A delete and recycle of the node that lands anywhere inside the
    /// call makes it return the old node's value, read before the delete,
    /// or panic as stale; it never returns the new occupant's value and
    /// never runs a closure against the occupant.
    pub fn get<T: Value>(&self, handle: Incr<T>) -> T {
        if handle.runtime_id() != self.runtime_id {
            return self.read_mismatch(handle);
        }
        let slot = handle.slot();
        let node = self.nodes.get(slot);
        self.park(ParkPoint::GetAfterHandleCheck(NodeId(slot)));

        // The raw byte, so the fast path skips `from_u8`'s range check.
        let raw = C::state_load_acquire(node.state_cell());
        if raw == state::CLEAN_BYTE {
            let (type_tag, arena_slot) = self.fields_under(handle, node);
            let type_tag = self.dep_stack.record_dep(
                self.runtime_id,
                NodeId(slot),
                handle.generation(),
                type_tag,
            );
            return match self.try_read_slot(handle, node, type_tag, arena_slot) {
                Some(value) => value,
                None => match self.reload_after_empty_slot(handle, node, type_tag, arena_slot) {
                    Ok(value) => value,
                    Err(now) => self.get_slow(handle, now),
                },
            };
        }
        self.get_slow(handle, NodeState::from_u8(raw))
    }

    /// The non-Clean arm of [`Self::get`]: walk, then read. `observed`
    /// is the state `get` loaded.
    #[inline(never)]
    fn get_slow<T: Value>(&self, handle: Incr<T>, observed: NodeState) -> T {
        match self.read_slow(handle, observed, ReadKind::Get) {
            Ok(value) => value,
            Err(cycle) => self.raise_cycle(cycle),
        }
    }

    /// Read the current value of a node, like [`Self::get`], or the cycle
    /// that leaves it without one: a node that landed
    /// [`NodeState::Cyclic`], or a read that closes a cycle through a
    /// query with no cycle strategy. Nodes with a value, a cycle
    /// strategy's included, read `Ok`.
    pub fn try_get<T: Value>(&self, handle: Incr<T>) -> Result<T, Cycle> {
        if handle.runtime_id() != self.runtime_id {
            return Ok(self.read_mismatch(handle));
        }
        let slot = handle.slot();
        let node = self.nodes.get(slot);
        // The raw byte, so the fast path skips `from_u8`'s range check.
        let raw = C::state_load_acquire(node.state_cell());
        if raw == state::CLEAN_BYTE {
            let (type_tag, arena_slot) = self.fields_under(handle, node);
            let type_tag = self.dep_stack.record_dep(
                self.runtime_id,
                NodeId(slot),
                handle.generation(),
                type_tag,
            );
            return match self.try_read_slot(handle, node, type_tag, arena_slot) {
                Some(value) => Ok(value),
                None => match self.reload_after_empty_slot(handle, node, type_tag, arena_slot) {
                    Ok(value) => Ok(value),
                    Err(now) => self.try_get_slow(handle, now),
                },
            };
        }
        self.try_get_slow(handle, NodeState::from_u8(raw))
    }

    #[inline(never)]
    fn try_get_slow<T: Value>(&self, handle: Incr<T>, observed: NodeState) -> Result<T, Cycle> {
        self.read_slow(handle, observed, ReadKind::TryGet)
    }

    /// The cycle of the group `handle` landed in last, if its last
    /// landing was as a member of a cycle.
    pub fn cycle_of(&self, handle: impl Into<ErasedIncr>) -> Option<Cycle> {
        let handle = handle.into();
        if handle.runtime_id() != self.runtime_id {
            if handle.runtime_id() == RuntimeId::PENDING && !self.pending_in_reach(handle) {
                self.pending_out_of_reach();
            }
            return None;
        }
        let inner = self.inner.read();
        if self.nodes.get(handle.slot()).generation() != handle.generation() {
            return None;
        }
        inner
            .member_recs
            .get(&handle.slot())
            .map(|group| group.cycle.clone())
    }

    /// The slow read shared by `get` and `try_get`: a cycle read, or a
    /// walk and then the slot.
    #[inline(always)]
    fn read_slow<T: Value>(
        &self,
        handle: Incr<T>,
        observed: NodeState,
        kind: ReadKind,
    ) -> Result<T, Cycle> {
        let slot = handle.slot();
        let node = self.nodes.get(slot);
        if node.generation() != handle.generation() {
            self.stale_handle(handle);
        }
        let frame = self
            .dep_stack
            .record_read(self.runtime_id, NodeId(slot), handle.generation());

        let word = value_code::thread_word();
        if word
            & (value_code::COUNTER_MASK
                | value_code::REPLAY_BIT
                | value_code::UNWIND_MASK
                | value_code::FOREIGN_BIT)
            != 0
            || frame.is_some_and(|f| self.doomed_or_past_budget(f))
        {
            return self.read_cold(handle, observed, kind, frame, word);
        }
        self.read_walk::<T, 0>(handle, node, observed, kind, word, frame)
    }

    /// Whether a read from frame `f` must take the cold path: the frame
    /// is doomed, or it reads at or past the frame budget, or a byte
    /// budget is set.
    #[inline(always)]
    fn doomed_or_past_budget(&self, f: ActiveRead) -> bool {
        // SAFETY: the active frame's control block is live while it is on
        // the stack.
        unsafe { &*f.ctl }.has(cycle::DOOMED)
            || f.depth >= self.max_frames.load(std::sync::atomic::Ordering::Relaxed)
            || self
                .max_stack_bytes
                .load(std::sync::atomic::Ordering::Relaxed)
                != 0
    }

    /// The rest of a slow read: a cycle read, or a walk in `MODE` and
    /// then the slot.
    #[inline(always)]
    fn read_walk<T: Value, const MODE: u8>(
        &self,
        handle: Incr<T>,
        node: &NodeData<C>,
        mut observed: NodeState,
        kind: ReadKind,
        word: u64,
        frame: Option<ActiveRead>,
    ) -> Result<T, Cycle> {
        let slot = handle.slot();
        let _root = reserve::Root::arm(!C::SINGLE_THREADED, (self.runtime_id, slot));
        loop {
            if MODE & VIRT != 0 && matches!(observed, NodeState::New | NodeState::Dirty) {
                if let Some(value) = self.stashed_read(handle) {
                    return Ok(value);
                }
            }
            // A read whose walk a doom stopped goes on with it even if its
            // root is held by now, as the unbounded run's walk goes on past
            // the entry that left it held.
            if (observed == NodeState::Cyclic || is_computing(observed))
                && !(MODE & VIRT != 0 && self.resume_pending(NodeId(slot)))
            {
                let armed = frame.filter(|_| MODE & ARMED != 0);
                if let Some(read) = self.cycle_read(handle, observed, kind, armed) {
                    return read;
                }
            }
            // `get` found the node not Clean, so the walk starts at once.
            let walk =
                || self.clean_walk::<MODE>(NodeId(slot), handle.generation(), word, observed);
            match self.entered(kind, walk)? {
                Computed::Done | Computed::Unsettled => {
                    if MODE & VIRT != 0 {
                        if let Some(value) = self.stashed_read(handle) {
                            return Ok(value);
                        }
                    }
                }
                Computed::Gone => self.stale_handle(handle),
                Computed::Deferred => {
                    return self
                        .cycle_read(
                            handle,
                            NodeState::Computing,
                            kind,
                            frame.filter(|_| MODE & ARMED != 0),
                        )
                        .expect("incr-core: a deferred root is held");
                }
                Computed::Doomed | Computed::Discarded => {
                    return Ok(self.placeholder_read::<T>(frame));
                }
                Computed::Skipped(_) | Computed::Replan | Computed::Revisit => {
                    unreachable!(
                        "incr-core: a walk's root is never skipped, replanned or revisited"
                    )
                }
            }
            self.park(ParkPoint::AfterWalk(NodeId(slot)));
            let (type_tag, arena_slot) = self.fields_under(handle, node);
            match self.try_read_slot(handle, node, type_tag, arena_slot) {
                Some(value) => {
                    // Context reservations live only while a context is open.
                    let reserving = !C::SINGLE_THREADED
                        && value_code::thread_word() & value_code::REPLAY_BIT != 0
                        && reserve::any();
                    if (frame.is_some() || reserving)
                        && state::load::<C>(node.state_cell()) == NodeState::Dirty
                    {
                        if let Some(f) = frame {
                            if !self.unchanged_stash(handle.slot(), handle.generation()) {
                                self.note_dirty_read(f, handle, &value);
                            }
                        }
                        if reserving {
                            self.end_unowed_reservation(slot, handle.generation());
                        }
                    }
                    return Ok(value);
                }
                None => match self.read_missing(handle, node) {
                    Ok(cycle) => return Err(cycle),
                    Err(NodeState::Clean) => {
                        return Ok(self.read_slot(handle, node, type_tag, arena_slot));
                    }
                    Err(now) => observed = now,
                },
            }
        }
    }

    /// A read after a walk found no value: the node landed `Cyclic`, or
    /// New as a member without a value of an episode that landed Dirty.
    /// Under `Shared` another thread can claim or land the node between
    /// the walk and these loads, so `Err` hands back the state the read
    /// goes on from: a Clean one is read, any other walked again.
    #[cold]
    #[inline(never)]
    fn read_missing<T: Value>(
        &self,
        handle: Incr<T>,
        node: &NodeData<C>,
    ) -> Result<Cycle, NodeState> {
        let state = state::load::<C>(node.state_cell());
        if matches!(state, NodeState::Cyclic | NodeState::New) {
            if let Some(cycle) = self.cycle_of(handle) {
                return Ok(cycle);
            }
        }
        if C::SINGLE_THREADED {
            self.read_failed(handle);
        }
        Err(state)
    }

    /// [`Self::get_slow`] reached from value code, a trait method of a
    /// value, key or element type that read a node `get` did not find
    /// Clean. Value code must not compute or wait, so the read reloads
    /// the state and succeeds only if the node is Clean by now.
    #[cold]
    #[inline(never)]
    fn value_code_read<T: Value>(&self, handle: Incr<T>, node: &NodeData<C>) -> T {
        let id = NodeId(handle.slot());
        self.park(ParkPoint::ValueCodeBeforeReload(id));
        let mut spins = 0;
        let mut state = state::load::<C>(node.state_cell());
        loop {
            // A `claim_set` that rolls back leaves the node as it found it,
            // so the read waits for the outcome instead of failing on it.
            while state == NodeState::Deleting {
                spins += 1;
                self.wait_out_deleting(id, spins);
                state = state::load::<C>(node.state_cell());
            }
            if state != NodeState::Clean {
                break;
            }
            let (type_tag, arena_slot) = self.fields_under(handle, node);
            match self.try_read_slot(handle, node, type_tag, arena_slot) {
                Some(value) => return value,
                None => match self.reload_after_empty_slot(handle, node, type_tag, arena_slot) {
                    Ok(value) => return value,
                    Err(now) => state = now,
                },
            }
        }
        match state {
            // A concurrent delete is no contract breach, so the read gets
            // the panic any reader of a node being deleted gets. Under
            // Local only this thread can be mid-delete, so the value code
            // was called from inside the delete.
            NodeState::Dead if !C::SINGLE_THREADED => {
                if node.generation() != handle.generation() {
                    self.stale_handle(handle);
                }
                deleted_while_reading(id)
            }
            _ => value_code::contract_panic(),
        }
    }

    /// A node's arena tag and slot, loaded in seqlock order: the fields
    /// first, then the generation under an Acquire fence, so a generation
    /// that still matches the handle says the fields are this node's,
    /// not an occupant's (the recycle bumps the generation before its
    /// Release fence and the field stores, `reinit_recycled`).
    #[inline(always)]
    fn fields_under<T: Value>(&self, handle: Incr<T>, node: &NodeData<C>) -> (u16, u32) {
        let type_tag = node.type_tag();
        let arena_slot = node.arena_slot();
        C::fence_acquire();
        if node.generation() != handle.generation() {
            self.stale_handle(handle);
        }
        (type_tag, arena_slot)
    }

    /// Clone the value behind a node's tag and arena slot, both read
    /// under a generation that matched the handle. Takes no lock, so the
    /// clone, which is value code, runs outside every incr lock. The
    /// arena may have released or reassigned the slot since, so the value
    /// counts only if the generation is unmoved after the read: a recycle
    /// bumps the generation before it writes the slot, and a read that
    /// loads the occupant's value synchronizes with that write.
    #[inline(always)]
    fn read_slot<T: Value>(
        &self,
        handle: Incr<T>,
        node: &NodeData<C>,
        type_tag: u16,
        arena_slot: u32,
    ) -> T {
        match self.try_read_slot(handle, node, type_tag, arena_slot) {
            Some(value) => value,
            None => self.read_failed(handle),
        }
    }

    /// [`Self::read_slot`], with `None` for an empty slot.
    #[inline(always)]
    fn try_read_slot<T: Value>(
        &self,
        handle: Incr<T>,
        node: &NodeData<C>,
        type_tag: u16,
        arena_slot: u32,
    ) -> Option<T> {
        self.park(ParkPoint::GetBeforeRead(NodeId(handle.slot())));
        let arena = self.arenas.tagged_arena::<T>(type_tag);
        let value = value_code::run(|| arena.and_then(|arena| arena.try_read(arena_slot)));
        C::fence_acquire();
        if node.generation() != handle.generation() {
            self.stale_handle(handle);
        }
        value
    }

    /// Read the current value and return a propagation trace alongside.
    /// Records per-node events (Recomputed { value_changed } or
    /// VerifiedClean) for every compute or short-circuit that happens
    /// during this `get`.
    pub fn get_traced<T: Value>(&self, handle: Incr<T>) -> (T, crate::trace::PropagationTrace) {
        use crate::trace::TraceAction;

        // Arm tracing: clear any prior log, then flip the gate so
        // compute_one starts appending events.
        {
            let mut inner = self.inner.write();
            inner.trace_log.clear();
        }
        C::u8_store_release(&self.tracing_armed, 1);

        // Disarm on every exit path: a panicking compute inside the
        // traced get must not leave tracing armed forever.
        struct Disarm<'a, C2: Cells>(&'a Runtime<C2>);
        impl<C2: Cells> Drop for Disarm<'_, C2> {
            fn drop(&mut self) {
                C2::u8_store_release(&self.0.tracing_armed, 0);
            }
        }
        let disarm = Disarm(self);

        // std has no clock on wasm32-unknown-unknown and `Instant::now()`
        // panics there, so the trace reports no elapsed time.
        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        let (value, elapsed_ns) = {
            let start = std::time::Instant::now();
            let value = self.get(handle);
            (value, start.elapsed().as_nanos() as u64)
        };
        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        let (value, elapsed_ns) = (self.get(handle), 0);

        drop(disarm);
        let node_traces: Vec<crate::trace::NodeTrace> = {
            let mut inner = self.inner.write();
            std::mem::take(&mut inner.trace_log)
        };

        let nodes_recomputed = node_traces
            .iter()
            .filter(|t| matches!(t.action, TraceAction::Recomputed { .. }))
            .count();
        let nodes_cutoff = node_traces
            .iter()
            .filter(|t| {
                matches!(
                    t.action,
                    TraceAction::Recomputed {
                        value_changed: false
                    }
                )
            })
            .count();

        let trace = crate::trace::PropagationTrace {
            target: NodeId(handle.slot()),
            node_traces,
            total_nodes: self.node_count(),
            nodes_recomputed,
            nodes_cutoff,
            elapsed_ns,
        };
        (value, trace)
    }

    /// Number of live nodes in the runtime (deleted-and-not-yet-recycled
    /// slots excluded).
    pub fn node_count(&self) -> usize {
        let free = self.inner.read().free_nodes.len();
        self.nodes.len() as usize - free
    }

    /// How many dependent lists have outgrown their inline storage into a
    /// heap block. For memory tests.
    #[doc(hidden)]
    pub fn dependent_list_blocks(&self) -> usize {
        let inner = self.inner.read();
        (0..self.nodes.len() as usize)
            .filter(|&slot| inner.dependents[slot].has_block())
            .count()
    }

    /// Assign a human-readable label to a node slot. Surfaces in
    /// `graph_snapshot()` and trace output. Re-assigning replaces.
    pub fn set_label(&self, slot: u32, label: String) {
        self.inner.write().labels.insert(slot, label);
    }

    /// Retrieve the label for a node slot, if any.
    pub fn label(&self, slot: u32) -> Option<String> {
        self.inner.read().labels.get(&slot).cloned()
    }

    /// Structural snapshot of every node. Returns `NodeInfo` with each
    /// node's dependencies (read from inline-7 storage) and dependents
    /// (read from the inner state).
    pub fn graph_snapshot(&self) -> Vec<crate::trace::NodeInfo> {
        use crate::trace::{NodeInfo, NodeKindInfo};
        let inner = self.inner.read();
        let count = self.nodes.len();
        let mut out = Vec::with_capacity(count as usize);
        for slot in 0..count {
            let node = self.nodes.get(slot);
            if state::load::<C>(node.state_cell()) == NodeState::Dead {
                continue;
            }
            let kind = if inner
                .compute_fns
                .get(slot as usize)
                .is_some_and(|f| f.is_some())
            {
                NodeKindInfo::Compute
            } else {
                NodeKindInfo::Input
            };
            let mut dependencies = Vec::new();
            node.for_each_dep(|d| dependencies.push(d));
            let dependents = inner
                .dependents
                .get(slot as usize)
                .map(|l| l.iter().collect())
                .unwrap_or_default();
            out.push(NodeInfo {
                id: NodeId(slot),
                kind,
                label: inner.labels.get(&slot).cloned(),
                dependencies,
                dependents,
            });
        }
        out
    }

    /// Set a new value on an input node. Bumps revision and marks all
    /// transitive dependents dirty.
    ///
    /// Panics if the handle refers to a query (compute) node. Setting a
    /// query node would overwrite its computed value and bypass the
    /// state machine; the only valid setter is the compute closure itself.
    ///
    /// A `set` holds a claim on the input across its compare and its
    /// write, so a second set of the input waits for it and a delete
    /// waits for it before retiring the node; readers never see it. A
    /// `set` holder never waits while claimed. A delete and recycle that
    /// lands anywhere inside the call makes it panic as stale; it never
    /// writes into the occupant.
    pub fn set<T: Value>(&self, handle: Incr<T>, value: T) {
        value_code::check();
        self.check_runtime(handle);
        let (claim, arena, arena_slot) = self.claim_input(handle);
        let (displaced, _) = self.write_claimed(claim, arena, arena_slot, value, drop);
        // The old value's user `Drop`, or the reclaim its retire may run,
        // happens with no claim held and none of the locks this set took.
        drop(displaced);
        reclaim_probe::reached(RetireSite::SetRetire);
    }

    /// `handle`'s `changed_at`, loaded with Acquire. `u64::MAX` while a
    /// `set` of it is between its compare and its settle.
    pub(crate) fn changed_at<T: Value>(&self, handle: Incr<T>) -> u64 {
        self.check_handle(handle);
        self.nodes.get(handle.slot()).changed_at()
    }

    /// [`Self::changed_at`] of a handle that may be dead or recycled:
    /// `None` unless the node is alive at the handle's generation through
    /// the read.
    pub(crate) fn changed_at_erased(&self, handle: ErasedIncr) -> Option<u64> {
        if !self.is_alive(handle) {
            return None;
        }
        let node = self.nodes.get(handle.slot());
        let stamp = node.changed_at();
        C::fence_acquire();
        (node.generation() == handle.generation()).then_some(stamp)
    }

    /// [`Self::set`] that hands back what it would drop: the displaced
    /// value if it wrote, or the rejected equal value if it did not, with
    /// the input's `changed_at` as this call left it.
    pub(crate) fn set_returning_displaced<T: Value>(
        &self,
        handle: Incr<T>,
        value: T,
    ) -> (Compared<T, Displaced<T, C>>, u64) {
        value_code::check();
        self.check_runtime(handle);
        let (claim, arena, arena_slot) = self.claim_input(handle);
        let mut rejected = None;
        let (displaced, stamp) =
            self.write_claimed(claim, arena, arena_slot, value, |r| rejected = Some(r));
        let compared = match rejected {
            Some(rejected) => Compared::Equal(rejected),
            None => Compared::Replaced(displaced),
        };
        (compared, stamp)
    }

    /// Take `set`'s claim on `handle`'s input, with the arena and slot
    /// its value lives in.
    fn claim_input<T: Value>(
        &self,
        handle: Incr<T>,
    ) -> (SetClaim<'_, C>, &GenericArena<T, C>, u32) {
        let slot = handle.slot();
        let id = NodeId(slot);
        let (node, flags) = self.nodes.get_with_flags(slot);
        self.park(ParkPoint::SetBeforeClaim(id));
        let mut spins: u32 = 0;
        loop {
            {
                // A recycle needs `inner.write()`, so under the hold the
                // generation compare is exact, the fields read are this
                // node's, and the claim cannot land on an occupant: a
                // delete marks the node before it unhooks it.
                let inner = self.inner.read();
                if node.generation() != handle.generation() {
                    drop(inner);
                    self.stale_handle(handle);
                }
                let is_query = inner
                    .compute_fns
                    .get(slot as usize)
                    .is_some_and(|f| f.is_some());
                assert!(
                    !is_query,
                    "Runtime::set called on a query (compute) node at slot {}; only input nodes can be set",
                    slot,
                );
                // Looked up before the claim so that its panic cannot
                // leave the claim held with no guard to release it.
                let arena = self
                    .arenas
                    .tagged_arena::<T>(node.type_tag())
                    .expect("incr-core: arena missing for input handle's type");
                match node.try_claim_set() {
                    Ok(base) => {
                        let arena_slot = node.arena_slot();
                        drop(inner);
                        set_stack::push(self.runtime_id, slot);
                        let claim = SetClaim {
                            rt: self,
                            node,
                            flags,
                            id,
                            base,
                            held: true,
                        };
                        self.park(ParkPoint::SetAfterClaim(id));
                        return (claim, arena, arena_slot);
                    }
                    Err(word) if word & NodeData::<C>::DELETED != 0 => {
                        drop(inner);
                        if C::SINGLE_THREADED {
                            value_code::contract_panic();
                        }
                        self.stale_handle(handle);
                    }
                    Err(_) => {}
                }
            }
            // Another set holds the input. It takes no lock before its
            // release, so this waits with none held.
            self.refuse_wait_on_own_set(id);
            self.spin_on(id, &mut spins, || !node.set_claimed());
        }
    }

    /// `set` from its claim to its walk. Returns what the write
    /// displaced, if it wrote, and the input's `changed_at` read under the
    /// claim: the stamp this write settled, or the one an equal value
    /// already carried. An equal value is handed to `rejected` once its
    /// earlier change has walked.
    fn write_claimed<T: Value>(
        &self,
        claim: SetClaim<'_, C>,
        arena: &GenericArena<T, C>,
        arena_slot: u32,
        value: T,
        rejected: impl FnOnce(T),
    ) -> (Option<Displaced<T, C>>, u64) {
        let node = claim.node;
        // Publication order is the protocol's backbone:
        // 1. pending marker, so no verifier can stamp past this change
        //    while the concrete revision is still unknown; it runs
        //    between the compare and the swap, so an equal value or a
        //    panicking `PartialEq` marks nothing;
        // 2. value swap, so any thread that later observes the bumped
        //    revision also observes the new value (swap is sequenced
        //    before the AcqRel fetch_add);
        // 3. revision bump;
        // 4. settle changed_at to the revision's input stamp;
        // 5. claim release, then the dirty walk, whose Release stores
        //    publish all of the above to claimants that Acquire the
        //    Dirty state.
        match arena.compare_and_write(arena_slot, value, || node.mark_changed_pending()) {
            Compared::Equal(equal) => {
                // The set that wrote this value released its claim before
                // its walk, and this call must not return before that
                // change is visible to its caller's dependents.
                let changed_at = node.changed_at();
                drop(claim);
                // The stamp is the odd one below its set's revision, and
                // `walked` is even, so reaching the stamp reaches the set.
                self.wait_walked(changed_at);
                rejected(equal);
                (None, changed_at)
            }
            Compared::Replaced(displaced) => {
                let stamp = claim.finish_written();
                (displaced, stamp)
            }
        }
    }
    /// Spin until revision `rev`'s walk is recorded. Under `Local` the
    /// watermark is the revision, so this never spins.
    fn wait_walked(&self, rev: u64) {
        debug_assert_ne!(rev, NodeData::<C>::CHANGED_PENDING);
        let mut spins: u32 = 0;
        while C::walked(&self.walk, &self.revision) < rev {
            spins += 1;
            C::backoff(spins);
        }
    }

    /// One backoff on `id`, found `Computing` by a reader. Another thread
    /// owns the compute. Bounded spin first (computes are usually short),
    /// then yield. Local never waits here: single-threaded Computing is a
    /// cycle or a broken invariant.
    /// Past `REGISTER_AFTER` spins the wait registers for wound-wait.
    #[inline]
    fn wait_on_claim(&self, id: NodeId, spins: u32) {
        if self.holds_claim(id) {
            self.waits_on_own_claim(id);
        }
        self.refuse_wait_on_computing(id);
        if spins == 1 {
            crate::lock_order::assert_may_wait_on_claim();
        }
        if !C::SINGLE_THREADED && spins >= waits::REGISTER_AFTER {
            let node = self.nodes.get(id.0);
            self.wait_registered(id, None, || {
                !is_computing(state::load::<C>(node.state_cell()))
            });
        } else {
            C::backoff(spins);
        }
    }

    /// A wait reached a claim this thread holds, which no wait releases.
    /// Reads resolve such nodes through the cycle rules before they wait,
    /// so only a path with no read to answer gets here.
    #[cold]
    #[inline(never)]
    fn waits_on_own_claim(&self, id: NodeId) -> ! {
        panic!(
            "incr-core: dependency cycle detected: node {} was reached while \
             its own compute was in progress",
            id.0,
        );
    }

    /// Whether this thread holds `id`'s compute claim: a frame computing
    /// it, or a held claim.
    pub(crate) fn holds_claim(&self, id: NodeId) -> bool {
        self.dep_stack.is_computing(self.runtime_id, id)
            || cycle_state::holds(self.runtime_id, id.0)
    }

    /// Before waiting on `id`'s `Computing` holder: under `Local` no
    /// other thread exists to release a claim, so a `Computing` node that
    /// is not on this runtime's frames is an invariant broken by user
    /// code.
    fn refuse_wait_on_computing(&self, id: NodeId) {
        if C::SINGLE_THREADED {
            panic!(
                "incr-core: internal invariant violated: node {} is Computing but no compute \
                 of it is in progress on this runtime; a Drop, Clone or PartialEq that calls \
                 into the runtime is the likely cause",
                id.0,
            );
        }
    }

    /// Before waiting on a set's claim on input `id`: a claim this thread
    /// holds is never released by waiting, and under `Local` every claim
    /// is this thread's. Only a value's `Drop` can get here under its own
    /// set's claim (`PartialEq` is value code), when the `PartialEq` panic
    /// it unwinds from drops the incoming value.
    fn refuse_wait_on_own_set(&self, id: NodeId) {
        if C::SINGLE_THREADED || set_stack::holds(self.runtime_id, id.0) {
            reached_under_own_set(id);
        }
    }

    /// Before a delete claims input `id`: refused here, while nothing is
    /// claimed, because a delete that marked the node first would leave
    /// it `Dead` under a set claim only this thread could release. The
    /// claim bit is tested first so that the thread-local scan runs only
    /// when some set holds the input: this thread's own claim is always
    /// visible to it.
    fn refuse_delete_under_own_set(&self, id: NodeId) {
        if self.nodes.get(id.0).set_claimed() && set_stack::holds(self.runtime_id, id.0) {
            reached_under_own_set(id);
        }
    }

    /// BFS forward walk from `start`'s dependents, marking each Clean
    /// node as Dirty. Every transition is a CAS so the walk can never
    /// clobber a `Computing` claim it raced with. `start_flags` is the
    /// set input's flags byte, so that a watched input is queued too.
    fn mark_dependents_dirty(&self, start: NodeId, start_flags: &C::U8) {
        let mut queue: Vec<NodeId> = Vec::new();
        {
            let inner = self.inner.read();
            if C::u8_load_relaxed(start_flags) & WATCHED != 0 {
                self.enqueue_watched(start, self.nodes.get(start.0));
            }
            let first = &inner.dependents[start.0 as usize];
            // Sized up front so a short walk allocates once and never
            // takes the grow path; a node nothing reads allocates nothing.
            if !first.is_empty() {
                queue = Vec::with_capacity(8);
                first.extend_into(&mut queue);
            }
        }

        let mut expanded_new: Option<HashSet<NodeId>> = None;
        while let Some(id) = queue.pop() {
            let (node, flags) = self.nodes.get_with_flags(id.0);
            loop {
                let cur = state::load::<C>(node.state_cell());
                match cur {
                    NodeState::Clean => {
                        if state::try_transition::<C>(node.state_cell(), cur, NodeState::Dirty)
                            .is_ok()
                        {
                            let inner = self.inner.read();
                            if C::u8_load_relaxed(flags) & WATCHED != 0 {
                                self.enqueue_watched(id, node);
                            }
                            inner.dependents[id.0 as usize].extend_into(&mut queue);
                            break;
                        }
                        // The state moved under us (a claim or another
                        // walk); re-examine.
                    }
                    NodeState::Failed => {
                        if self.flip_failed(id) {
                            let inner = self.inner.read();
                            if C::u8_load_relaxed(flags) & WATCHED != 0 {
                                self.enqueue_watched(id, node);
                            }
                            inner.dependents[id.0 as usize].extend_into(&mut queue);
                            break;
                        }
                        // The state moved under us (a claim or another
                        // walk); re-examine.
                    }
                    NodeState::Computing => {
                        // An in-flight compute may have read the old
                        // input value and will stamp timestamps that
                        // predate this set. Flag it so its finishing CAS
                        // fails and it lands Dirty, and walk through to
                        // its dependents, which would otherwise stay
                        // Clean against a stale parent.
                        if state::try_transition::<C>(
                            node.state_cell(),
                            NodeState::Computing,
                            NodeState::ComputingDirty,
                        )
                        .is_ok()
                        {
                            let inner = self.inner.read();
                            if C::u8_load_relaxed(flags) & WATCHED != 0 {
                                self.enqueue_watched(id, node);
                            }
                            inner.dependents[id.0 as usize].extend_into(&mut queue);
                            break;
                        }
                    }
                    NodeState::New => {
                        // A node that recovered from a stale dep lands New
                        // but keeps its old reverse edges, so a Failed
                        // dependent can only be reached through it.
                        // Never-run queries have no dependents. The visit
                        // is an RMW for the reason given on the Dirty arm.
                        // Former cycle members are New with edges on each
                        // other, so each is expanded once per walk.
                        if state::try_transition::<C>(node.state_cell(), cur, cur).is_ok() {
                            let inner = self.inner.read();
                            let list = &inner.dependents[id.0 as usize];
                            if !list.is_empty()
                                && expanded_new.get_or_insert_with(HashSet::default).insert(id)
                            {
                                list.extend_into(&mut queue);
                            }
                            break;
                        }
                    }
                    NodeState::Cyclic => {
                        // The cycle may break with this set, so its members
                        // go back to New, which recomputes them.
                        if state::try_transition::<C>(node.state_cell(), cur, NodeState::New)
                            .is_ok()
                        {
                            let inner = self.inner.read();
                            if C::u8_load_relaxed(flags) & WATCHED != 0 {
                                self.enqueue_watched(id, node);
                            }
                            inner.dependents[id.0 as usize].extend_into(&mut queue);
                            break;
                        }
                    }
                    NodeState::Dirty | NodeState::ComputingDirty => {
                        // Already flagged; the walk that flagged it also
                        // covered its dependents. The visit must still be
                        // an RMW, not a plain load: this walk stores a
                        // dep's Dirty and then reaches the node, while a
                        // verifier claims the node and then loads its deps'
                        // states. With a plain load here that is the Dekker
                        // pattern, and both sides may read the old value.
                        // As an RMW the visit sits in the node's
                        // modification order, so the verifier's claim either
                        // reads from it (and sees the dep Dirty) or this CAS
                        // fails on the claim and the next pass flags it.
                        if state::try_transition::<C>(node.state_cell(), cur, cur).is_ok() {
                            break;
                        }
                    }
                    NodeState::Dead => {
                        // Raced a delete through a not-yet-unhooked
                        // edge; a deleted node has no dependents to
                        // propagate to.
                        break;
                    }
                    NodeState::Deleting => {
                        // The claim lands Dead or rolls back to the state
                        // it took, which this walk must still flag.
                        self.wait_out_deleting(id, 1);
                    }
                }
            }
        }
    }

    /// Ensure the node at `id` is Clean, recomputing the minimum
    /// necessary subgraph.
    ///
    /// Iterative post-order walk over the non-Clean subgraph. A node is
    /// expanded on every pop until the walk has made `EXACT_MODE_AT` first
    /// visits; from then on each node is expanded at most once (exact
    /// mode), so a cycle of Dirty nodes, whose recorded edges re-expand
    /// the ring forever, costs one pass over the ring past the switch
    /// instead of a traversal budget. Cycles themselves are caught in
    /// `compute_one` via the frame stack.
    ///
    /// A non-root dep found Computing on this thread's own held set is a
    /// stale recorded edge or a real cycle. The walk skips it and taints
    /// the path to it; a tainted node is verified but never recomputed by
    /// the walk, so only a closure that still reads it, and so re-enters
    /// the held node through a nested `get`, reports a cycle. The root is
    /// exempt because its caller reads its slot as soon as the walk
    /// returns.
    ///
    /// `gen` is the caller's handle generation for the root. A walk entry
    /// carries none (installed deps are ids only); one whose slot is
    /// recycled under the walk is dropped, because the parent that
    /// recorded it sees the dead dep at its publish. A root found
    /// recycled is reported as `Gone` for the caller's stale panic.
    #[inline]
    fn ensure_clean(&self, id: NodeId, gen: u32) -> Computed {
        // The check precedes the fast path because a Clean read still
        // takes locks in this release.
        let word = value_code::checked_word();

        // Fast path: already clean.
        let observed = state::load::<C>(self.nodes.get(id.0).state_cell());
        if observed == NodeState::Clean {
            return Computed::Done;
        }
        self.clean_walk::<0>(id, gen, word, observed)
    }

    /// [`Self::ensure_clean`] past its fast path, for a caller that took
    /// `word` from `value_code::checked_word` and loaded `id` as
    /// `observed`, not Clean. A root that went Clean since is left at its
    /// first pop, or after one backoff if it was loaded Computing.
    ///
    /// A node found Computing is waited on here once before its pending
    /// entry's compute loads it again, so that no Computing state is
    /// loaded twice before the first backoff.
    fn clean_walk<const MODE: u8>(
        &self,
        id: NodeId,
        gen: u32,
        word: u64,
        observed: NodeState,
    ) -> Computed {
        // A read whose walk a doom stopped goes on with it even if its root
        // has been memoized since, as the unbounded run's walk goes on.
        let resume = if MODE & VIRT != 0 {
            self.take_resume(id)
        } else {
            None
        };
        if resume.is_none()
            && (word != 0 && taint_memo::memoized(word, self.runtime_id, id)
                || MODE & VIRT != 0 && self.take_direct(id))
        {
            if is_computing(observed) {
                self.wait_on_claim(id, 1);
            }
            return self.compute_direct::<MODE>(id, gen);
        }
        // A resumed walk was planned before its doom, at a count no longer
        // known, so it checks against none: it re-plans if any other
        // thread has landed an episode.
        let mut planned = if resume.is_some() {
            Planned { all: 0, own: 0 }
        } else {
            self.planned()
        };
        let mut walk = resume.unwrap_or_else(|| Walk::new(id, observed));
        let mut root = Computed::Done;
        let mut unchecked = None;
        #[cfg(test)]
        let mut revisited: Vec<NodeId> = Vec::new();

        while let Some(Entry { id: cur, taint }) = walk.work.pop() {
            let is_root = walk.work.is_empty();
            if taint < VISIT_COMPUTING {
                self.park(ParkPoint::BeforeComputeOne(cur));
                let visit = if is_root || taint >= NO_TAINT {
                    Visit::Full
                } else {
                    Visit::VerifyOnly(taint)
                };
                // A tainted entry never runs its closure, so it verifies in
                // place whatever another thread landed; a tainted root runs
                // it, but its taint has replaced the New mark.
                #[cfg(test)]
                let forced = *FORCE_REVISIT && !revisited.contains(&cur);
                #[cfg(not(test))]
                let forced = false;
                let seen = if C::SINGLE_THREADED || taint < NO_TAINT {
                    Seen::Unwalked
                } else if taint == NO_TAINT_NEW || forced {
                    Seen::New
                } else {
                    Seen::Dirty
                };
                let mut computed = self.compute_one::<MODE>(
                    cur,
                    visit,
                    is_root.then_some(gen),
                    !is_root,
                    (!C::SINGLE_THREADED && !is_root && unchecked.take() != Some(cur))
                        .then_some(planned),
                    seen,
                );
                if !C::SINGLE_THREADED && computed == Computed::Revisit {
                    #[cfg(test)]
                    revisited.push(cur);
                    walk.revisit(cur);
                    continue;
                }
                if !C::SINGLE_THREADED && computed == Computed::Replan {
                    planned = self.planned();
                    if !self.prune_settled(&mut walk) {
                        walk.push(Entry { id: cur, taint });
                        unchecked = Some(cur);
                    }
                    continue;
                }
                if computed != Computed::Done {
                    let entry = Entry { id: cur, taint };
                    match self.walk_stopped(id, entry, visit, is_root, &mut walk, computed) {
                        Ok(after) => computed = after,
                        Err(stopped) => return stopped,
                    }
                }
                if is_root {
                    root = computed;
                } else if taint < NO_TAINT {
                    walk.taint(taint);
                }
                continue;
            }
            #[cfg(test)]
            self.walk_pops
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if walk.is_visited(cur) {
                continue;
            }

            let (node, flags) = self.nodes.get_with_flags(cur.0);
            let observed = if taint == VISIT {
                state::load::<C>(node.state_cell())
            } else {
                NodeState::Computing
            };
            // A node with a virtual frame stands for a frame on the stack,
            // so the walk skips it as it skips a real one.
            if MODE & VIRT != 0
                && !is_root
                && matches!(
                    observed,
                    NodeState::New
                        | NodeState::Dirty
                        | NodeState::Computing
                        | NodeState::ComputingDirty
                )
            {
                if let Some(owner) = self.virtual_owner(cur) {
                    walk.taint(owner);
                    continue;
                }
            }
            match observed {
                NodeState::Clean | NodeState::Cyclic => continue,
                NodeState::Dirty
                    if C::u8_load_relaxed(flags) & IN_GROUP != 0
                        && self.expand_group(cur, &mut walk) =>
                {
                    continue
                }
                NodeState::Computing | NodeState::ComputingDirty => {
                    if !is_root {
                        if let Some(owner) = self.own_claim_owner(cur) {
                            #[cfg(all(test, not(loom)))]
                            if self.is_virtual_cidx(owner - 1) {
                                self.virtual_anchor_skips
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            walk.taint(owner);
                            continue;
                        }
                    }
                    // Another thread owns this subtree and is cleaning
                    // its deps itself; expanding them here would tear
                    // through a dep list mid-rewrite. Wait for the owner,
                    // then visit the node again: a claim released without
                    // a Clean landing (a wound's discard, a flagged
                    // landing) leaves deps the walk has not expanded, and
                    // one of them may be a claim this thread holds, which
                    // only the expansion skips.
                    self.park(ParkPoint::BeforeComputeOne(cur));
                    let mut spins = 0;
                    loop {
                        spins += 1;
                        self.wait_on_claim(cur, spins);
                        if !is_computing(state::load::<C>(node.state_cell())) {
                            break;
                        }
                    }
                    walk.push(Entry {
                        id: cur,
                        taint: VISIT,
                    });
                }
                NodeState::Deleting => {
                    self.wait_out_deleting(cur, 1);
                    walk.push(Entry {
                        id: cur,
                        taint: VISIT,
                    });
                }
                _ => {
                    // New nodes can carry installed edges on each other
                    // after a cycle breaks, so each is expanded at most
                    // once per walk. A never-run node has no deps and
                    // never touches the set.
                    if observed == NodeState::New
                        && node.dep_count() != 0
                        && !walk.first_new_expansion(cur)
                    {
                        continue;
                    }
                    #[cfg(test)]
                    self.expansions
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    walk.visit(cur, observed == NodeState::New);
                    node.for_each_dep(|dep| {
                        let dep_node = self.nodes.get(dep.0);
                        let dep_state = state::load::<C>(dep_node.state_cell());
                        if !dep_state.is_settled() {
                            walk.expand(dep, dep_state);
                        }
                    });
                }
            }
        }

        #[cfg(test)]
        self.first_visits.fetch_add(
            u64::from(walk.first_visits),
            std::sync::atomic::Ordering::Relaxed,
        );
        root
    }

    /// What a walk under `Shared` plans under: the runtime's episode
    /// count and this thread's share of it.
    #[inline(always)]
    fn planned(&self) -> Planned {
        if C::SINGLE_THREADED {
            return Planned { all: 0, own: 0 };
        }
        let all = C::u64_load_acquire(&self.episodes_landed);
        // This thread's share is at most the count.
        let own = if all == 0 {
            0
        } else {
            cycle_state::own_episodes(self.runtime_id)
        };
        Planned { all, own }
    }

    /// Whether another thread has landed an episode on this runtime since
    /// `planned`.
    #[inline(always)]
    fn replan(&self, planned: Planned) -> bool {
        #[cfg(test)]
        if *FORCE_REPLAN || FORCE_REPLAN_HERE.with(std::cell::Cell::get) {
            return true;
        }
        let all = C::u64_load_acquire(&self.episodes_landed);
        all != planned.all && self.others_landed(planned, all)
    }

    #[cold]
    #[inline(never)]
    fn others_landed(&self, planned: Planned, all: u64) -> bool {
        let own = cycle_state::own_episodes(self.runtime_id);
        all.wrapping_sub(planned.all) != own.wrapping_sub(planned.own)
    }

    /// After a claim CAS of `id` at `generation` succeeded: whether another
    /// thread's doom reserved the node, so the claim must be given back
    /// unrun and the reservation waited out.
    #[inline(always)]
    fn reserved_elsewhere(&self, id: NodeId, generation: u32) -> bool {
        !C::SINGLE_THREADED
            && reserve::any()
            && reserve::elsewhere((self.runtime_id, id.0), generation)
    }

    /// Releases a claim taken from New or Dirty (`dirty`) before anything
    /// ran, as no landing: watchers see nothing. A set's flag changes
    /// nothing, since its walk already passed through to the dependents and
    /// leaves a New node New.
    #[cold]
    #[inline(never)]
    fn unclaim_unrun(&self, node: &NodeData<C>, dirty: bool) {
        #[cfg(test)]
        self.unrun_unclaims
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let back = if dirty {
            NodeState::Dirty
        } else {
            NodeState::New
        };
        node.set_unrun(true);
        if state::try_transition::<C>(node.state_cell(), NodeState::Computing, back).is_err() {
            let unflagged =
                state::try_transition::<C>(node.state_cell(), NodeState::ComputingDirty, back);
            debug_assert!(unflagged.is_ok(), "incr-core: an unrun claim is held");
        }
    }

    /// Drops the part of `walk`'s plan under its lowest pending node that
    /// is Clean or Cyclic now, as a walk planned now would not expand it,
    /// and keeps the rest with what the walk has learned. False if no
    /// pending node has settled.
    #[cold]
    #[inline(never)]
    fn prune_settled(&self, walk: &mut Walk) -> bool {
        let settled = walk.work.iter().position(|e| {
            e.is_pending()
                && matches!(
                    state::load::<C>(self.nodes.get(e.id.0).state_cell()),
                    NodeState::Clean | NodeState::Cyclic
                )
        });
        match settled {
            Some(at) => {
                walk.prune(at + 1);
                true
            }
            None => false,
        }
    }

    /// The first time a walk reaches a Dirty member of a landed group, it
    /// expands the whole group: a pending entry for the member, whose
    /// compute verifies the group, and every member's deps outside the
    /// group. Edges between members are never pushed, so the group costs
    /// one expansion per walk. Returns false for a stale `IN_GROUP` bit,
    /// whose node is expanded as an ordinary Dirty node.
    #[cold]
    #[inline(never)]
    fn expand_group(&self, cur: NodeId, walk: &mut Walk) -> bool {
        let group = self.inner.read().member_recs.get(&cur.0).cloned();
        let Some(group) = group else {
            return false;
        };
        let generation = self.nodes.get(cur.0).generation();
        if !group
            .members
            .iter()
            .any(|&(m, g)| m == cur.0 && g == generation)
            || cycle_state::is_dissolved_group(self.runtime_id, cur.0, &group)
        {
            return false;
        }
        let groups = &mut walk.expanded.get_or_insert_with(Box::default).groups;
        if groups.iter().any(|g| Arc::ptr_eq(g, &group)) {
            return true;
        }
        groups.push(group.clone());
        #[cfg(test)]
        self.expansions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        walk.visit(cur, false);
        for &(m, _) in &group.members {
            self.nodes.get(m).for_each_dep(|dep| {
                if group
                    .members
                    .binary_search_by_key(&dep.0, |&(s, _)| s)
                    .is_ok()
                {
                    return;
                }
                let dep_state = state::load::<C>(self.nodes.get(dep.0).state_cell());
                if !dep_state.is_settled() {
                    walk.expand(dep, dep_state);
                }
            });
        }
        true
    }

    /// A root the taint memo holds, computed without a walk. Out of line
    /// so that the walk's own inlined `compute_one` never tests for
    /// `Visit::Direct`.
    #[cold]
    #[inline(never)]
    fn compute_direct<const MODE: u8>(&self, id: NodeId, gen: u32) -> Computed {
        let mut computed =
            self.compute_one::<MODE>(id, Visit::Direct, Some(gen), false, None, Seen::Unwalked);
        if computed == Computed::Discarded {
            computed = self.walk_discarded(Visit::Direct).unwrap_or(computed);
        }
        if computed == Computed::Unsettled {
            return Computed::Done;
        }
        computed
    }

    /// Compute (or verify) a single node, assuming all its known deps
    /// are already clean. Handles state-machine transitions and red/green
    /// early cutoff.
    ///
    /// Protocol invariants (see the shared-read-protocol-v2 decision):
    /// - every transition is a CAS; nothing ever clobbers a Computing
    ///   claim;
    /// - a thread observing Computing waits for the owner, unless the
    ///   node is on its own frame stack, which is a dependency cycle;
    /// - stamps use the revision captured BEFORE verify/compute via
    ///   fetch_max, so a concurrent set() invalidates rather than masks.
    ///
    /// `gen` is the root's handle generation, `None` for a walk entry.
    /// The closure and the generation `g0` are read together under the
    /// read hold, and a claim counts only while the generation is still
    /// `g0` after the CAS (which reads from the recycle's Release store):
    /// a delete and recycle between the hold and the claim would
    /// otherwise run the dead node's closure against the occupant, and a
    /// root compares `g0` with `gen` first, since the recycle may have
    /// landed before the hold.
    ///
    /// The call owns one [`Deferred`] bag for every attempt it makes (a
    /// Stale retry carries it), and every exit that returns drains it
    /// after that exit's landing CAS and poison, the exits that land
    /// nothing included (`Gone`, the input arm, a retry that found the
    /// node computed by another thread). The `Err` path drains in
    /// carried mode before it resumes the panic.
    ///
    /// `planned` is what a walk entry under `Shared` was planned under, and
    /// `seen` what the walk's visit found of an untainted entry.
    #[inline(always)]
    fn compute_one<const MODE: u8>(
        &self,
        id: NodeId,
        visit: Visit,
        gen: Option<u32>,
        walked: bool,
        planned: Option<Planned>,
        seen: Seen,
    ) -> Computed {
        let mut bag = Deferred::new();
        let computed =
            self.compute_attempts::<MODE>(id, visit, gen, walked, planned, seen, &mut bag);
        bag.drain(self);
        computed
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn compute_attempts<const MODE: u8>(
        &self,
        id: NodeId,
        visit: Visit,
        gen: Option<u32>,
        walked: bool,
        planned: Option<Planned>,
        seen: Seen,
        bag: &mut Deferred<C>,
    ) -> Computed {
        let (node, flags) = self.nodes.get_with_flags(id.0);
        let mut spins: u32 = 0;
        let mut g0 = gen;

        'attempt: loop {
            let observed = state::load::<C>(node.state_cell());
            // A wait can outlast the node, whose slot then holds another.
            if spins != 0 && g0.is_some_and(|g| g != node.generation()) {
                return Computed::Gone;
            }
            // After the load, so a landing that passed the held payload's
            // failure on is seen whatever state it left the node in since.
            if MODE & VIRT != 0 && !C::SINGLE_THREADED && !walked {
                if let Some(payload) = self.owed_payload(id, g0.unwrap_or(node.generation())) {
                    self.raise_owed(payload, bag);
                }
            }
            match observed {
                NodeState::Clean | NodeState::Cyclic => return Computed::Done,
                NodeState::Computing | NodeState::ComputingDirty => {
                    // A walk's deps-first computes can defer the walk's own
                    // root into this thread's held map: its reader reads it
                    // by the cycle rules, as a deferred compute's.
                    if spins == 0 && cycle_state::holds(self.runtime_id, id.0) {
                        return Computed::Deferred;
                    }
                    if spins == 0 {
                        self.park(ParkPoint::ComputeWaits(id));
                    }
                    spins += 1;
                    self.wait_on_claim(id, spins);
                    continue;
                }
                NodeState::Deleting => {
                    spins += 1;
                    self.wait_out_deleting(id, spins);
                    continue;
                }
                // A walk's entry leaves a Failed dep for the reading node,
                // whose rerun reads it only if it still needs it.
                NodeState::Failed if walked => return Computed::Done,
                NodeState::Failed => {
                    self.read_failed_node(id);
                    continue;
                }
                NodeState::Dead => {
                    // A bumped generation under Dead is a delete that has
                    // unhooked the node but not recycled the slot yet.
                    if g0.is_some_and(|g| g != node.generation()) {
                        return Computed::Gone;
                    }
                    // An observer's read of a node whose delete has not
                    // bumped its generation yet: the observer goes with it.
                    if observe_impl::is_observed(self.runtime_id, id.0) {
                        return Computed::Gone;
                    }
                    deleted_while_reading(id);
                }
                NodeState::New => {
                    // Nothing to verify against, and the walk must not run
                    // the closure of a tainted node.
                    if let Visit::VerifyOnly(owner) = visit {
                        self.memoize(id, owner);
                        return Computed::Done;
                    }
                    // The thread waited on ran the node and left it New with
                    // the deps it read, which a later read expands first.
                    if seen != Seen::Unwalked
                        && spins != 0
                        && g0.is_none_or(|g| g == node.generation())
                        && !node.unrun()
                    {
                        #[cfg(test)]
                        self.wait_revisits
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Computed::Revisit;
                    }
                }
                // The generation check below would end a recycled root.
                NodeState::Dirty
                    if seen != Seen::Unwalked
                        && spins != 0
                        && g0.is_none_or(|g| g == node.generation())
                        // A claim released unrun leaves its thread's read
                        // going on, which may still come after this one.
                        && !node.unrun() =>
                {
                    #[cfg(test)]
                    self.wait_revisits
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Computed::Revisit;
                }
                NodeState::Dirty => {}
            }
            if MODE & VIRT != 0 && spins != 0 && gen.is_none() {
                if let Some(owner) = self.post_wait_owner(id) {
                    return Computed::Skipped(owner);
                }
            }

            // Distinguish input vs query: inputs don't compute, they
            // just need their state stamped clean.
            let (compute, g) = {
                let inner = self.inner.read();
                let compute = inner
                    .compute_fns
                    .get(id.0 as usize)
                    .and_then(|f| f.as_deref())
                    .map(std::ptr::from_ref);
                (compute, node.generation())
            };
            if g0.is_some_and(|g0| g0 != g) {
                return Computed::Gone;
            }
            g0 = Some(g);
            let compute = match compute {
                Some(f) => f,
                None => {
                    node.max_verified_at(self.current_revision());
                    if state::try_transition::<C>(node.state_cell(), observed, NodeState::Clean)
                        .is_ok()
                    {
                        return Computed::Done;
                    }
                    continue;
                }
            };

            if MODE & VIRT != 0 && self.stashed_claim(id, g) {
                return Computed::Done;
            }
            self.park(ParkPoint::BeforeClaim(id));

            // Verification claims the node too, so a walk that arrives
            // meanwhile flags ComputingDirty and blocks the Clean landing.
            // The claim source tells us exactly whether this is a
            // recompute: the node may have gone Dirty since `observed`.
            let mut verifying = observed == NodeState::Dirty && visit != Visit::Direct;
            let claimed_dirty = if verifying {
                if state::try_transition::<C>(
                    node.state_cell(),
                    NodeState::Dirty,
                    NodeState::Computing,
                )
                .is_err()
                {
                    continue;
                }
                true
            } else {
                match state::try_claim_compute::<C>(node.state_cell()) {
                    Ok(src) => src == NodeState::Dirty,
                    Err(_) => continue,
                }
            };

            if node.generation() != g {
                // The claim landed on the slot's next occupant, so leave
                // it as found: New from New, Dirty otherwise, because a
                // computed occupant must keep its early cutoff.
                let discard = if claimed_dirty {
                    NodeState::Dirty
                } else {
                    NodeState::New
                };
                self.land_unrun(id, node, discard);
                return Computed::Gone;
            }
            self.park(ParkPoint::BeforeReservationCheck(id));
            if self.reserved_elsewhere(id, g) {
                self.unclaim_unrun(node, claimed_dirty);
                self.wait_out_reservation(id, g, None, &mut spins);
                continue;
            }
            // The claim succeeds on a Dirty that such a landing and a set
            // made since the load, and until it is released no landing of
            // the node can pass the failure on.
            if MODE & VIRT != 0 && !C::SINGLE_THREADED && !walked {
                if let Some(payload) = self.owed_payload(id, g) {
                    let discard = if claimed_dirty {
                        NodeState::Dirty
                    } else {
                        NodeState::New
                    };
                    self.land_unrun(id, node, discard);
                    self.raise_owed(payload, bag);
                }
            }
            if claimed_dirty && seen == Seen::New {
                self.unclaim_unrun(node, claimed_dirty);
                return Computed::Revisit;
            }
            // When the claim reads from another thread's episode, which left
            // this node New or Dirty, it reads the count that episode
            // bumped first: the episode may have landed Clean a node this
            // plan reaches the node through, which a walk planned now
            // skips. A claim not ordered after an episode may run before it.
            if let Some(p) = planned.filter(|_| !C::SINGLE_THREADED) {
                self.park(ParkPoint::AfterWalkClaim(id));
                if self.replan(p) {
                    self.unclaim_unrun(node, claimed_dirty);
                    return Computed::Replan;
                }
            }

            // Capture the revisions after the claim and BEFORE verifying or
            // computing. If a set() bumps the revision mid-flight, our
            // stamps stay below its changed_at and the next reader
            // re-verifies; stamping the post-compute revision would mask
            // the change forever. Before the claim is too early: the claim
            // can succeed on a later Dirty than `observed` (another reader
            // cleans the node, a second set dirties it again), and a change
            // stamped below that set hides it from dependents verified in
            // between. The claim reads from the walk that made the node
            // Dirty, so the loads below see that walk's revision bump.
            //
            // changed_at uses the current revision, so a value changed by
            // an in-flight set is never under-stamped. verified_at uses the
            // walked watermark instead: a set bumps the revision before its
            // walk, and a verified_at naming a revision whose walk had not
            // dirtied our deps yet would let us verify Clean past it (B6).
            let stamp_rev = self.current_revision();
            let mut verify_rev = C::walked(&self.walk, &self.revision);

            let mut adopted_panic = None;
            if C::u32_load_relaxed(&self.stash_count) != 0 && !matches!(visit, Visit::VerifyOnly(_))
            {
                // A group member's verify is its group's, which no stash
                // stands for.
                if verifying
                    && C::u8_load_relaxed(self.nodes.get_with_flags(id.0).1) & IN_GROUP != 0
                {
                    self.drop_stash(id, g, bag);
                } else {
                    match self.adopt_stash(id, g, claimed_dirty, walked, bag) {
                        replay_impl::Adopted::No => {}
                        replay_impl::Adopted::Panicked(raised) => {
                            verifying = false;
                            adopted_panic = Some(raised);
                        }
                        replay_impl::Adopted::Landed(computed) => {
                            self.adopted_pop(id);
                            return computed;
                        }
                        // The stashed run ended, as its pop does, before its
                        // landing found a dep gone; the retry is the next run.
                        replay_impl::Adopted::Stale => {
                            self.adopted_pop(id);
                            self.park_with_bag(ParkPoint::StaleRetry(id), bag);
                            continue 'attempt;
                        }
                    }
                }
            }

            // The flags byte is loaded only now, after the claim and the
            // generation recheck: until the claim, another thread's
            // episode can land the node into a group or a single-node
            // landing can take it out of one.
            if verifying && C::u8_load_relaxed(flags) & IN_GROUP != 0 {
                // Group verify claims the other members with no anchor the
                // doom's discards would release, so a read that dooms skips
                // it and dooms as if the verify failed. A tainted visit's
                // group verify runs no member and must not doom, because the
                // unbounded run's verify leaves the node for its reader.
                if MODE & ARMED != 0 && !matches!(visit, Visit::VerifyOnly(_)) {
                    return self.doom(
                        id,
                        node,
                        g,
                        claimed_dirty,
                        matches!(visit, Visit::Direct),
                        walked,
                    );
                }
                if MODE & VIRT != 0
                    && !matches!(visit, Visit::VerifyOnly(_))
                    && replay::speculative()
                {
                    return self.drop_speculative_verify(id, node, claimed_dirty);
                }
                // A speculative tainted verify leaves the node Dirty for its
                // reader as a failed one would, but keeps the group, whose
                // release or re-landing would outlive the run's discard.
                if let Visit::VerifyOnly(owner) = visit {
                    if MODE & VIRT != 0 && replay::speculative() {
                        self.land_unrun(id, node, NodeState::Dirty);
                        self.memoize(id, owner);
                        return Computed::Done;
                    }
                }
                match self.group_verify(id, g, visit, walked, bag) {
                    // A tainted verify leaves the node Dirty for the root's
                    // compute to read again, which poisons where needed.
                    GroupEnd::Done(Computed::Done) if !matches!(visit, Visit::VerifyOnly(_)) => {
                        self.park(ParkPoint::EpisodeLanded(id));
                        return self.episode_landed(node, walked);
                    }
                    GroupEnd::Done(computed) => return computed,
                    GroupEnd::Retry => {
                        if MODE & VIRT != 0 {
                            self.rerun_root(id, self.dep_stack.next_cidx());
                        }
                        self.park_with_bag(ParkPoint::StaleRetry(id), bag);
                        continue 'attempt;
                    }
                    // A member could not be claimed, or the group is no
                    // longer a cycle: compute, and rediscover any cycle
                    // that remains.
                    GroupEnd::Ordinary => verifying = false,
                    GroupEnd::Dissolved => {
                        if MODE & VIRT != 0 {
                            self.rerun_root(id, self.dep_stack.next_cidx());
                        }
                        verifying = false;
                    }
                }
            }

            // Red/green verification, only from Dirty: a New node has
            // nothing to verify against. It counts a dep that is not Clean
            // as changed, which catches a walk that passed this node before
            // the claim, and a dep that landed Dirty without a new stamp.
            // A `verified_at` of 0 means never verified, and an empty slot
            // may sit behind it, so it always fails.
            if verifying {
                let my_verified = node.verified_at();
                let mut any_changed = my_verified == 0;
                node.for_each_dep(|dep| {
                    if !any_changed && !self.observed_clean(dep, my_verified) {
                        any_changed = true;
                    }
                });
                if !any_changed {
                    self.park(ParkPoint::VerifyAfterScan(id));
                    if MODE & VIRT != 0 && replay::speculative() {
                        return self.stash_verify(id, node, g, my_verified, stamp_rev, verify_rev);
                    }
                    node.max_verified_at(verify_rev);
                    let landed =
                        self.land_single(id, node, flags, NodeState::Clean, NodeState::Dirty);
                    self.end_reservation_landed(id, node, landed);
                    if landed == NodeState::Clean {
                        self.record_trace(id, crate::trace::TraceAction::VerifiedClean);
                    } else {
                        self.poison_reader(walked);
                    }
                    return Computed::Done;
                }
                if let Visit::VerifyOnly(owner) = visit {
                    // The root's own compute re-reads this node through a
                    // nested get if it still needs it, and that read
                    // poisons where needed; a poison here would cost the
                    // root a Clean landing for an edge it may have dropped.
                    self.land_single_as(id, node, flags, NodeState::Dirty, NodeState::Dirty, true);
                    self.memoize(id, owner);
                    return Computed::Done;
                }
                // Keep the claim and recompute; the closure's nested reads
                // clean the changed deps.
            }
            if MODE & ARMED != 0 && adopted_panic.is_none() {
                return self.doom(
                    id,
                    node,
                    g,
                    claimed_dirty,
                    matches!(visit, Visit::Direct),
                    walked,
                );
            }
            let speculative = MODE & VIRT != 0 && replay::speculative();
            if speculative && !self.replayable(id.0) && !self.mark_tried(id, g) {
                let target = if claimed_dirty {
                    NodeState::Dirty
                } else {
                    NodeState::New
                };
                self.land(id, node, target, target);
                return Computed::Done;
            }
            let is_recompute = claimed_dirty;
            let first = !claimed_dirty;

            'run: loop {
                // Track deps via the strategy's dep stack. The closure runs
                // under a panic boundary: a panicking user closure must not
                // strand the node in Computing (wedging every future reader)
                // or leak its dep frame. AssertUnwindSafe is justified
                // because the only state the closure can reach is the
                // runtime's, and the Failed transition plus the frame pop
                // below restore every invariant the unwind could have
                // interrupted; no lock is held while user code runs.
                let ctl = ComputeCtl::resolved_by(
                    if is_recompute { cycle::RECOMPUTE } else { 0 }
                        | if walked { cycle::WALKED } else { 0 }
                        | if speculative { cycle::SPECULATIVE } else { 0 },
                    self.keep_resolver_ptr(),
                );
                let cidx = self.dep_stack.push_frame(self.runtime_id, id, &ctl);
                ctl.lowlink.set(cidx);
                if MODE & VIRT != 0 {
                    self.start_run(id, cidx, &ctl);
                    if let Some(low) = self.take_low(id, cidx) {
                        ctl.lower(low);
                    }
                    // The unbounded run's frame claimed before the sets its
                    // doomed runs made, so a dep they changed supersedes it.
                    if let Some(rev) = self.take_claim_rev(id, cidx) {
                        verify_rev = verify_rev.min(rev);
                    }
                    self.divert_for_deliveries(id, cidx);
                }
                // SAFETY: the closure was read under the read hold at
                // generation `g`, and the claim above was rechecked against `g`.
                // A closure is freed only by `release_claimed`, under a Dead
                // claim and after the generation bump, and no Dead claim can
                // succeed while this node is Computing. `SlotVec` never moves
                // its entries, so the `Box` still points where it was read.
                let compute = unsafe { &*compute };
                let mut result = match adopted_panic.take() {
                    Some(raised) => Err(self.raise_adopted(raised)),
                    None => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        compute(self, id.0, &ctl)
                    })),
                };
                let (deps, reading) = self.dep_stack.pop_frame_reading();
                let word = value_code::thread_word();
                if word
                    & (value_code::MEMO_BIT
                        | value_code::UNWIND_MASK
                        | value_code::REPLAY_BIT
                        | value_code::NOTES_BIT
                        | value_code::KEPT_BIT)
                    != 0
                {
                    let attempt = Attempt {
                        first,
                        cidx,
                        stamp_rev,
                        verify_rev,
                    };
                    if let Err(payload) = &result {
                        if word & value_code::UNWIND_MASK != 0 {
                            cycle_impl::clear_caught_cycle(&**payload, &deps, reading);
                        }
                    }
                    if let Some(dropped) =
                        self.after_pop_word(id, word, &mut result, attempt, &ctl, &deps, bag)
                    {
                        return dropped;
                    }
                }

                if result.is_err() && ctl.has(cycle::SPECULATIVE) {
                    self.unbuild_speculative(&ctl, bag);
                }
                let outcome = match result {
                    Ok(outcome) => outcome,
                    // A doomed run reads placeholders, so a cycle it raises
                    // is no more the unbounded run's than a panic is: its
                    // retry raises the cycle again if it is real.
                    Err(payload)
                        if ctl.has(cycle::DOOMED)
                            && !matches!(cycle_state::unwind_kind(), Some(UnwindKind::Wound)) =>
                    {
                        cycle_state::take_unwind_kind();
                        let attempt = Attempt {
                            first,
                            cidx,
                            stamp_rev,
                            verify_rev,
                        };
                        return self.discard_doomed(id, attempt, &ctl, Some(payload), bag);
                    }
                    Err(payload) => {
                        return self.compute_panicked(
                            id,
                            node,
                            flags,
                            Attempt {
                                first,
                                cidx,
                                stamp_rev,
                                verify_rev,
                            },
                            &ctl,
                            &deps,
                            bag,
                            walked,
                            payload,
                        )
                    }
                };

                if ctl.flags.get() & (cycle::LOW | cycle::DOOMED | cycle::SPECULATIVE) != 0 {
                    let attempt = Attempt {
                        first,
                        cidx,
                        stamp_rev,
                        verify_rev,
                    };
                    if ctl.has(cycle::DOOMED) {
                        return self.discard_doomed(id, attempt, &ctl, None, bag);
                    }
                    if !ctl.has(cycle::LOW) {
                        return self.stash_speculative(id, g, attempt, &ctl, deps, bag);
                    }
                    match self.cycle_pop(id, g, cidx, first, &ctl, deps, bag) {
                        PopEnd::Deferred => return Computed::Deferred,
                        PopEnd::Landed => {
                            self.park(ParkPoint::EpisodeLanded(id));
                            return self.episode_landed(node, walked);
                        }
                        PopEnd::Suspended | PopEnd::Dropped => return Computed::Discarded,
                        PopEnd::Contained => return Computed::Unsettled,
                        PopEnd::Dissolved => {
                            if MODE & VIRT != 0 {
                                self.rerun_root(id, cidx);
                            }
                            self.end_dissolved(id, bag);
                            if MODE & VIRT != 0 {
                                self.end_dissolved_reads(id, g, cidx);
                            }
                            continue 'run;
                        }
                        PopEnd::Retry => {
                            if MODE & VIRT != 0 {
                                self.rerun_root(id, cidx);
                            }
                            self.park_with_bag(ParkPoint::StaleRetry(id), bag);
                            continue 'attempt;
                        }
                    }
                }
                debug_assert_ne!(outcome, Outcome::Buffered);
                let value_changed = outcome == Outcome::Changed;

                let published = self.publish_deps(id, &deps, None, verify_rev, bag);

                // The closure has already written its value into the slot, and
                // the next recompute's early cutoff compares against it, so a
                // changed value is stamped even when it is never returned.
                if value_changed || !is_recompute {
                    node.max_changed_at(stamp_rev);
                }

                if published == Publish::Stale {
                    // The value may have been read from a dep slot that was
                    // released or recycled under the closure, so it is never
                    // returned: land New and recompute. The retry either stops
                    // reading the dep or panics on its stale handle.
                    if first && ctl.wrote.get() == 0 {
                        node.set_verified_at(0);
                    }
                    // Leaving the group is safe: a New node's record is inert,
                    // and the retry computes the node alone.
                    self.land_run(
                        id,
                        node,
                        flags,
                        NodeState::New,
                        NodeState::New,
                        &ctl,
                        true,
                        bag,
                    );
                    self.park_with_bag(ParkPoint::StaleRetry(id), bag);
                    continue 'attempt;
                }

                let poisoned = ctl.has(cycle::POISONED);
                if poisoned {
                    // The value may be built on a result the closure
                    // swallowed from a dep that did not land Clean or whose
                    // cleanup failed, and that dep's stamps need not have
                    // moved, so a verified_at at or above them would let the
                    // next read verify this node Clean over that value.
                    // Zero fails verification, so the next read recomputes.
                    node.set_verified_at(0);
                } else {
                    node.max_verified_at(verify_rev);
                }

                // Finish. Land Dirty instead of Clean if a set() flagged us
                // ComputingDirty mid-compute, if a new dep changed after we
                // read it, or if a dep we read did not itself land Clean, so
                // the next reader recomputes against the newer input.
                let target = if published == Publish::Superseded || poisoned {
                    NodeState::Dirty
                } else {
                    NodeState::Clean
                };
                // The run computed the node alone, as the unbounded run's
                // does, so the node leaves its group.
                let landed =
                    self.land_run(id, node, flags, target, NodeState::Dirty, &ctl, false, bag);
                self.end_reservation_landed(id, node, landed);
                if landed != NodeState::Clean {
                    self.poison_reader(walked);
                    self.record_trace(id, crate::trace::TraceAction::Recomputed { value_changed });
                    return Computed::Unsettled;
                }
                self.record_trace(id, crate::trace::TraceAction::Recomputed { value_changed });
                return Computed::Done;
            }
        }
    }

    /// A root whose episode landed it: a reader that reads it Dirty, or
    /// New with no value, must not land Clean over it, as after a
    /// single-node landing. Under `Shared` another thread can claim the
    /// root as soon as it lands, so any state but Clean or Cyclic poisons:
    /// a Clean root leaves Clean only through a walk, which flags the
    /// reader too.
    fn episode_landed(&self, node: &NodeData<C>, walked: bool) -> Computed {
        if !matches!(node.state(), NodeState::Clean | NodeState::Cyclic) {
            self.poison_reader(walked);
        }
        Computed::Done
    }

    /// After a non-Clean landing: the enclosing compute read this node, and
    /// its own publish may take the fast path, where nothing else would
    /// catch that. A walk entry (`walked`) is no such read: a dependent's
    /// rerun reads it again through a nested get if it still needs it, and
    /// that read poisons the dependent.
    fn poison_reader(&self, walked: bool) {
        if !walked {
            self.dep_stack.poison_current_frame(self.runtime_id);
        }
    }

    /// The `Err` arm of a compute: the closure panicked, or an engine
    /// unwind passed through it. Lands the node and every claim this
    /// thread held above its frame, then resumes the unwind. A walk's entry
    /// (`walked`) contains a panic, or a cycle whose claims all ran above
    /// it, and returns `Unsettled`: the recorder's rerun reads the node
    /// again if it still does, and the payload stays with the Failed node
    /// for that read.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn compute_panicked(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        flags: &C::U8,
        attempt: Attempt,
        ctl: &ComputeCtl,
        deps: &[(NodeId, u32)],
        bag: &mut Deferred<C>,
        walked: bool,
        payload: Box<dyn std::any::Any + Send>,
    ) -> Computed {
        let kind = cycle_state::take_unwind_kind();
        let (payload, children_stashed) = if kind.is_none() && ctl.has(cycle::SPECULATIVE) {
            self.stash_panic(id, ctl, deps, &attempt, payload)
        } else {
            (payload, false)
        };
        replay::unwind_past(self.runtime_id, attempt.cidx);
        // A closure that catches a speculative run's discard can't land a
        // value over it.
        let kind = replay::run_unwind_kind(kind, replay::speculative());
        let contain = walked && self.containable(id, attempt.cidx, kind, &*payload);
        self.release_unwound_from(
            attempt.cidx,
            id.0,
            kind,
            kind.is_some(),
            Some(&*payload),
            bag,
        );
        let wrote = ctl.wrote.get() != 0;
        if children_stashed {
            let first = attempt.first;
            let stashed = ChildLanding::Stashed;
            self.discard_land_in_group(id, first, wrote, attempt.stamp_rev, stashed, bag);
        } else if kind.is_some() {
            // A cycle or a wound is no fault of this node's: it lands as
            // if it never ran, never Failed.
            self.unwound_land(id, attempt.first, wrote, attempt.stamp_rev, kind, bag);
        } else {
            // Publish the union of the old deps and whatever was recorded
            // before the panic. Without the recorded subset, a first-compute
            // panic leaves no reverse edges at all and no set() can ever
            // flip the node out of Failed; without the old set, a recompute
            // panic would lose recovery paths through deps the failed run
            // had not reached yet. Extra edges only cause spurious
            // re-verification, never staleness.
            self.land_failed(
                id,
                node,
                flags,
                attempt,
                wrote,
                deps,
                &[],
                Some(&*payload),
                bag,
            );
            self.end_reservation(id, node);
        }
        // After the landings, whose notes the kept copy carries.
        if !contain {
            self.note_unwound_read(&*payload, kind);
        }
        if contain {
            self.contain(attempt.cidx, payload, bag);
            return Computed::Unsettled;
        }
        self.leave_frame(kind, payload, bag)
    }

    /// The unwind out of a frame whose claims have landed goes on to the
    /// reading frame: a closure that catches it has read a dep that did not
    /// land Clean.
    #[cold]
    pub(super) fn leave_frame(
        &self,
        kind: Option<UnwindKind>,
        payload: Box<dyn std::any::Any + Send>,
        bag: &mut Deferred<C>,
    ) -> ! {
        self.dep_stack.poison_current_frame(self.runtime_id);
        cycle_state::set_unwind_kind(kind);
        bag.drain_carried(payload)
    }

    /// Whether walk entry `entry`, at position `cidx`, contains an unwind
    /// out of its compute, asked before the compute's frame releases its
    /// claims: a panic, or a cycle none of whose members this thread will
    /// still hold or has a virtual frame of, so that every member ran above
    /// the walk and the unwind released it. The entry's own claim and the
    /// claims held at or above its position are not counted: the unwind
    /// releases them too, and a virtual frame's run still holds the entry's
    /// when this is asked. A closure that swallowed a cycle and then
    /// panicked is contained as any panic. A wounded thread must reach its
    /// outermost entry, so it contains none.
    pub(super) fn containable(
        &self,
        entry: NodeId,
        cidx: u32,
        kind: Option<UnwindKind>,
        payload: &(dyn std::any::Any + Send),
    ) -> bool {
        if cycle_state::wounded() {
            return false;
        }
        match kind {
            None => true,
            Some(UnwindKind::Cycle) => payload.downcast_ref::<CycleUnwind>().is_none_or(|c| {
                c.0.participants()
                    .iter()
                    .map(|p| NodeId(p.slot()))
                    .all(|p| {
                        (p == entry || self.unwind_releases(p, cidx))
                            && self.virtual_owner(p).is_none()
                    })
            }),
            Some(_) => false,
        }
    }

    /// Fix 5's landing of a claim whose closure panicked: the union of
    /// the old and recorded deps is published, and the node lands Failed,
    /// or Dirty or New where publication says the panic may hide a change,
    /// or Dirty where a replay delivery raised the panic and its first take
    /// landed the node Dirty (`retaken_landing`).
    /// `co_members`, sorted, are the claims released with it, which count
    /// as settled. `payload` is the panic, when the caller has it.
    #[allow(clippy::too_many_arguments)]
    fn land_failed(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        flags: &C::U8,
        attempt: Attempt,
        wrote: bool,
        deps: &[(NodeId, u32)],
        co_members: &[u32],
        payload: Option<&(dyn std::any::Any + Send)>,
        bag: &mut Deferred<C>,
    ) {
        let landing = match self.publish_deps(id, deps, Some(co_members), attempt.verify_rev, bag) {
            // A stale dep's slot may already belong to another node, so no
            // union is published. The old deps keep their reverse edges.
            Publish::Stale => NodeState::New,
            // A new dep had a change in flight when we published, and that
            // change's walk may have read the dep's dependents before our
            // edge existed. Nothing would ever flip us out of Failed, so
            // land Dirty and let the next reader retry.
            Publish::Superseded => NodeState::Dirty,
            // verified_at was not raised, so once a dependency changes, the
            // dirty walk flips Failed -> Dirty and red/green sees the change
            // and recomputes for real. If a set() already flagged us
            // mid-compute, land Dirty now so the next reader retries.
            Publish::Published => NodeState::Failed,
        };
        let landing =
            self.retaken_landing(id, node.generation(), attempt.cidx, deps, payload, landing);
        let mut mark = 0;
        if landing == NodeState::Failed {
            if !C::SINGLE_THREADED {
                self.before_failed_landing(id, node.generation(), attempt.cidx, payload);
            }
            mark = self.failed_landing(id, node.generation(), attempt.cidx, payload, bag);
        }
        let flagged = if landing == NodeState::New {
            NodeState::New
        } else {
            NodeState::Dirty
        };
        // The slot already holds the new value when the panic came after
        // the write (a displaced value's Drop, or a reclaim inside its
        // retire). Stamp it: a recompute that produces the same value cuts
        // off, and without the stamp a dependent verified against the old
        // value would stay Clean over the new one. A first compute that
        // wrote nothing has an empty slot, so it must never verify.
        if wrote {
            node.max_changed_at(attempt.stamp_rev);
        } else if attempt.first {
            node.set_verified_at(0);
        }
        // After the publish, so that a child the attempt read keeps the
        // union's edge and is retained rather than deleted under it.
        self.children_landing(id, flags, ChildLanding::Discard, false, bag);
        // The panic fails this attempt in the unbounded run too, so the node
        // leaves its group.
        let landed = self.land_single(id, node, flags, landing, flagged);
        // A set flagged the compute, so it landed Dirty past the walk that
        // would have released its ticket.
        if mark != 0 && landed != NodeState::Failed {
            self.left_failed_landing(id, mark, bag);
        }
    }

    /// The landing of a claim a user panic released without failing it.
    /// A node that lands Failed, now or as the unwind goes on, read it,
    /// directly or through other released claims, so a set that would cure
    /// the failure may reach that node only through this one: the union of
    /// its old and recorded deps is published, as [`Self::land_failed`]
    /// publishes them, and it lands New, which a set's walk passes through
    /// where Dirty stops it.
    /// `co_members`, sorted, are the claims released with it.
    fn land_panic_released(
        &self,
        id: NodeId,
        attempt: Attempt,
        wrote: bool,
        deps: &[(NodeId, u32)],
        co_members: &[u32],
        bag: &mut Deferred<C>,
    ) {
        let (node, flags) = self.nodes.get_with_flags(id.0);
        // Any outcome lands New, which recomputes.
        self.publish_deps(id, deps, Some(co_members), attempt.verify_rev, bag);
        if wrote {
            node.max_changed_at(attempt.stamp_rev);
        } else if attempt.first {
            node.set_verified_at(0);
        }
        self.children_landing(id, flags, ChildLanding::Discard, false, bag);
        self.land_single(id, node, flags, NodeState::New, NodeState::New);
    }

    /// A frame pop found the memo, unwind or replay bit in the thread word.
    /// A wound or discard rewrites `result` into its unwind; `Some` is a
    /// speculative run dropped in place.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn after_pop_word(
        &self,
        id: NodeId,
        word: u64,
        result: &mut std::thread::Result<Outcome>,
        attempt: Attempt,
        ctl: &ComputeCtl,
        deps: &[(NodeId, u32)],
        bag: &mut Deferred<C>,
    ) -> Option<Computed> {
        // A doomed run's pop is discarded and its retry pops in its place,
        // so it retires nothing the unbounded run keeps.
        let doomed = ctl.has(cycle::DOOMED);
        if word & value_code::NOTES_BIT != 0 && result.is_ok() && !doomed {
            self.retire_notes_above(attempt.cidx);
        }
        if word & value_code::KEPT_BIT != 0 {
            replay::drop_kept_from(self.runtime_id, attempt.cidx);
        }
        if self.after_pop(word, result.is_ok()) {
            *result = Err(match UnwindKind::of_word(word) {
                Some(UnwindKind::Discard) => Box::new(cycle::DiscardUnwind),
                _ => Box::new(cycle::WoundUnwind),
            });
        } else if result.is_ok() && self.speculative_lands_dirty(id, word, ctl, deps) {
            if self.drop_cooperatively(None) {
                return Some(self.discard_dropped(id, attempt, ctl, bag));
            }
            self.set_drop_kind();
            *result = Err(Box::new(cycle::DiscardUnwind));
        }
        // After the rewrites: a swallowed discard or wound is no pop of the
        // unbounded run's.
        if result.is_ok()
            && !doomed
            && word & (value_code::NOTES_BIT | value_code::REPLAY_BIT | value_code::KEPT_BIT) != 0
        {
            self.retire_copies(&replay::Retire::Above(attempt.cidx));
        }
        // An engine unwind does not end the run: the retry after it is the
        // same run of the unbounded run's, and reads the cycle again from
        // its delivery.
        if word & value_code::REPLAY_BIT != 0
            && !doomed
            && (result.is_ok() || cycle_state::unwind_kind().is_none())
        {
            self.end_run(id, attempt.cidx);
        }
        None
    }

    /// A frame pop found the memo bit or an unwind kind in the thread
    /// word. A kind left set after a closure returned normally is an
    /// engine unwind the closure caught and swallowed; true if it was a
    /// wound, which the pop lands as an unwind.
    #[cold]
    #[inline(never)]
    fn after_pop(&self, word: u64, returned: bool) -> bool {
        if word & value_code::MEMO_BIT != 0 {
            taint_memo::clear_from(self.dep_stack.next_cidx() + 1);
        }
        returned
            && !std::thread::panicking()
            && self.swallowed(cycle_state::UnwindKind::of_word(word))
    }

    /// A single-node landing: a node landing outside an episode leaves
    /// its group, so its `MemberRec` is removed before the landing CAS,
    /// while the claim is still held. Once the CAS lands the node another
    /// thread can claim it and write a new record.
    #[inline(always)]
    pub(crate) fn land_single(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        flags: &C::U8,
        target: NodeState,
        flagged: NodeState,
    ) -> NodeState {
        self.land_single_as(id, node, flags, target, flagged, false)
    }

    #[inline(always)]
    pub(crate) fn land_single_as(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        flags: &C::U8,
        target: NodeState,
        flagged: NodeState,
        unrun: bool,
    ) -> NodeState {
        if C::u8_load_relaxed(flags) & IN_GROUP != 0 {
            self.leave_group(id, node, flags, target);
        }
        self.land_as(id, node, target, flagged, unrun)
    }

    /// The landing of a run of `id` in `compute_one`: a creator's
    /// children and a group the node leaves are settled first, while the
    /// claim is held, behind one flag test that only creators and former
    /// group members pass. `discard` says the run's attempt is discarded.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn land_run(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        flags: &C::U8,
        target: NodeState,
        flagged: NodeState,
        run: &ComputeCtl,
        discard: bool,
        bag: &mut Deferred<C>,
    ) -> NodeState {
        if C::u8_load_relaxed(flags) & (IN_GROUP | HAS_INDEX) != 0 {
            self.settle_before_landing(id, flags, target, run, discard, bag);
        }
        self.land(id, node, target, flagged)
    }

    #[cold]
    #[inline(never)]
    fn settle_before_landing(
        &self,
        id: NodeId,
        flags: &C::U8,
        target: NodeState,
        run: &ComputeCtl,
        discard: bool,
        bag: &mut Deferred<C>,
    ) {
        let bits = C::u8_load_relaxed(flags);
        if bits & HAS_INDEX != 0 {
            // A doomed frame's misses created nothing and its run may have
            // been cut short, so it keeps what it did not touch.
            let landing = if discard {
                ChildLanding::Discard
            } else if target == NodeState::Clean && !run.has(cycle::DOOMED) {
                ChildLanding::Sweep
            } else {
                ChildLanding::Keep
            };
            self.land_children(id, landing, !run.children.get().is_null(), bag);
        }
        if bits & IN_GROUP != 0 {
            self.leave_group(id, self.nodes.get(id.0), flags, target);
        }
    }

    /// Takes `id` out of its group for a landing in `target`. A member's
    /// value is its group's, which its own closure need not give, so one
    /// landing Dirty is left unverified (rule (b)) and reruns rather than
    /// verify on its own deps.
    #[cold]
    #[inline(never)]
    pub(crate) fn leave_group(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        flags: &C::U8,
        target: NodeState,
    ) {
        if target == NodeState::Dirty {
            node.set_verified_at(0);
        }
        let mut inner = self.inner.write();
        inner.member_recs.remove(&id.0);
        C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) & !IN_GROUP);
    }

    /// The taint-memo owner for a skip of `id`, if this thread (or, under
    /// `Local`, this runtime) holds its `Computing` claim: one more than
    /// the position of the frame computing it, or of the anchor of its
    /// held claim. This is the walk's own held set; a set's claim is not
    /// on `state`, so it never shows up here.
    fn own_claim_owner(&self, id: NodeId) -> Option<u32> {
        self.dep_stack
            .frame_cidx(self.runtime_id, id)
            .or_else(|| cycle_state::held_anchor(self.runtime_id, id.0))
            .map(|cidx| cidx + 1)
    }

    /// Memoize a node verify-only left non-Clean under the frame at depth
    /// `owner`, whose pop clears the entry. A speculative run memoizes
    /// nothing: the unbounded run has no such run, and an entry it left
    /// under an owner below it would send a later real read of the node
    /// straight to its compute.
    fn memoize(&self, id: NodeId, owner: u32) {
        if !replay::speculative() {
            taint_memo::memoize(self.runtime_id, id, owner);
        }
    }

    /// Leave this thread's `Computing` claim for `target`, or for `flagged`
    /// if a walk raised `ComputingDirty` meanwhile, and return the state
    /// landed. Both moves are CASes, never a plain store: a failed CAS is
    /// only a load and need not read the newest walker's flag, and a
    /// plain store would end the walkers' release sequence. The second
    /// CAS, as an RMW, reads the newest value, so every walker's earlier
    /// writes happen-before the next claim. Loom and Miri both treat a
    /// failed CAS as reading the newest value, so neither would catch a
    /// regression here.
    ///
    /// A landing off the Clean path, neither Clean nor Cyclic, then
    /// reports a watched node: the walk stops at a node already Dirty,
    /// so nothing else would.
    #[inline(always)]
    fn land(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        target: NodeState,
        flagged: NodeState,
    ) -> NodeState {
        self.land_as(id, node, target, flagged, false)
    }

    /// Ends this thread's context reservation of `id` if its run `landed`
    /// settled, as the unbounded run's frame releases its claim there. An
    /// unsettled landing keeps it until a later run of this context lands
    /// the node settled or the context is removed, so that no other thread
    /// reruns the node in between.
    pub(super) fn end_reservation_landed(&self, id: NodeId, node: &NodeData<C>, landed: NodeState) {
        if landed.is_settled() {
            self.end_reservation(id, node);
        }
    }

    /// Ends this thread's context reservation of `id`, whose run no retry
    /// follows. The context's untaken deliveries of the node's outcome
    /// stay: each stands for a read the unbounded frame made at this
    /// landing, and a later set lands that reader Dirty by its claim
    /// watermark.
    pub(super) fn end_reservation(&self, id: NodeId, node: &NodeData<C>) {
        if C::SINGLE_THREADED || !reserve::any() || replay::speculative() {
            return;
        }
        reserve::end_landed((self.runtime_id, id.0), node.generation());
    }

    /// [`Self::land`] of a claim released as if its run never happened, so
    /// that a thread that waited it out does not take it for a landing
    /// that a read ordered before its own made.
    pub(crate) fn land_unrun(&self, id: NodeId, node: &NodeData<C>, target: NodeState) {
        self.land_as(id, node, target, target, true);
    }

    #[inline(always)]
    fn land_as(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        target: NodeState,
        flagged: NodeState,
        unrun: bool,
    ) -> NodeState {
        node.set_unrun(unrun);
        if !C::SINGLE_THREADED && unrun {
            reserve::note((self.runtime_id, id.0), node.generation());
        }
        if !C::SINGLE_THREADED && !unrun && target == NodeState::Dirty {
            reserve::note_root((self.runtime_id, id.0), node.generation());
        }
        if state::try_transition::<C>(node.state_cell(), NodeState::Computing, target).is_ok() {
            if !matches!(target, NodeState::Clean | NodeState::Cyclic) {
                self.report_landing(id);
            }
            return target;
        }
        let owned =
            state::try_transition::<C>(node.state_cell(), NodeState::ComputingDirty, flagged);
        debug_assert!(
            owned.is_ok(),
            "only the claim's owner leaves ComputingDirty"
        );
        if !matches!(flagged, NodeState::Clean | NodeState::Cyclic) {
            self.report_landing(id);
        }
        flagged
    }

    /// Validate a compute's recorded deps, then install them and update
    /// reverse edges. With `panicked` the installed list is the union of
    /// the old deps and the recorded ones, and a new dep that is `Failed`
    /// does not supersede: the compute that read it fails with it, and the
    /// dep's own recovery walk reaches this node through the new edge. Diffs old vs
    /// new deps so static-dep queries (the common case) skip the
    /// inner.write() acquire and the dependents vector edits on
    /// recompute.
    ///
    /// The fast path needs no validation: its deps already have reverse
    /// edges, so a strict delete refuses them and a dirty walk reaches
    /// this node. The slow path validates, installs and diffs inside one
    /// `inner.write()` hold. That ordering is part of the contract: a
    /// delete claims `Dead` under `inner.read()`, and a set's walk flips a
    /// dep before reading its dependents under `inner.read()`, so each
    /// either finishes before validation sees it or sees this node's new
    /// edge.
    fn publish_deps(
        &self,
        id: NodeId,
        recorded: &[(NodeId, u32)],
        panic: Option<&[u32]>,
        verify_rev: u64,
        bag: &mut Deferred<C>,
    ) -> Publish {
        let node = self.nodes.get(id.0);
        let panicked = panic.is_some();

        // Compared in place so the static-dep fast path, the common case,
        // allocates nothing; under Local an allocation pair per compute
        // was the bulk of propagation cost.
        if !panicked {
            let mut seen = 0;
            let mut same = true;
            node.for_each_dep(|d| {
                same = same && recorded.get(seen).is_some_and(|&(r, _)| r == d);
                seen += 1;
            });
            if same && seen == recorded.len() {
                return Publish::Published;
            }
        }

        let mut old_deps: Vec<NodeId> = Vec::with_capacity(8);
        node.for_each_dep(|d| old_deps.push(d));
        let new_deps: Vec<NodeId> = if panicked {
            dep_set::union(&old_deps, recorded)
        } else {
            recorded.iter().map(|&(d, _)| d).collect()
        };

        if old_deps == new_deps {
            return Publish::Published;
        }
        if dep_set::needs_hashing(old_deps.len(), new_deps.len()) {
            return self.publish_large_diff(
                id, node, recorded, panic, verify_rev, bag, &old_deps, &new_deps,
            );
        }
        let diff = DepDiff {
            old: &old_deps,
            new: &new_deps,
            old_set: &old_deps[..],
            new_set: &new_deps[..],
        };
        self.publish_diff(id, node, recorded, panic, verify_rev, bag, diff)
    }

    /// `publish_deps` for lists long enough that scanning one per entry
    /// of the other would be quadratic. Out of line so the short-list
    /// path compiles to plain scans.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn publish_large_diff(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        recorded: &[(NodeId, u32)],
        panic: Option<&[u32]>,
        verify_rev: u64,
        bag: &mut Deferred<C>,
        old_deps: &[NodeId],
        new_deps: &[NodeId],
    ) -> Publish {
        // Built before the write hold, which every reader of the graph waits on.
        let old_set = DepHashSet::new(old_deps);
        let new_set = DepHashSet::new(new_deps);
        let diff = DepDiff {
            old: old_deps,
            new: new_deps,
            old_set: &old_set,
            new_set: &new_set,
        };
        self.publish_diff(id, node, recorded, panic, verify_rev, bag, diff)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn publish_diff<S: DepLookup + ?Sized>(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        recorded: &[(NodeId, u32)],
        panic: Option<&[u32]>,
        verify_rev: u64,
        bag: &mut Deferred<C>,
        diff: DepDiff<'_, S>,
    ) -> Publish {
        let mut inner = self.inner.write();
        let stale = recorded.iter().any(|&(d, generation)| {
            let dep = self.nodes.get(d.0);
            dep.generation() != generation || dep.state() == NodeState::Dead
        });
        if stale {
            return Publish::Stale;
        }
        // Against the walked watermark, not the stamp revision: a set
        // bumped but not yet walked at the claim can't reach this node
        // through an edge it lacks, and a dep recomputed for that set
        // lands Clean with a changed_at up to the stamp revision.
        let settled = |d: &NodeId| {
            diff.old_set.has(*d)
                || self.observed_clean(*d, verify_rev)
                || panic.is_some_and(|members| {
                    self.nodes.get(d.0).state() == NodeState::Failed
                        || members.binary_search(&d.0).is_ok()
                })
        };
        let superseded = !diff.new.iter().all(settled);

        // The displaced list is retired at the attempt's drain, after its
        // landing: a retire can reclaim pending values and run their
        // `Drop` under this hold otherwise.
        if let Some(displaced) = node.install_deps(diff.new) {
            bag.push_dep_list(displaced);
        }

        for old_dep in diff.old {
            if !diff.new_set.has(*old_dep) {
                inner.dependents[old_dep.0 as usize].remove(id);
            }
        }
        // A dep read twice is in `new_deps` twice, and its list already
        // ends with `id` from the first push, so each list holds `id` at
        // most once and a delete's back-to-front remove finds it.
        for new_dep in diff.new {
            if !diff.old_set.has(*new_dep) {
                inner.dependents[new_dep.0 as usize].push_unique(id);
            }
        }
        if superseded {
            Publish::Superseded
        } else {
            Publish::Published
        }
    }

    /// True iff `dep` is `Clean` with `changed_at <= rev`. The state is
    /// loaded first because the Release that publishes a recompute's
    /// `changed_at` stamp is its finishing state CAS: reading the stamp
    /// first could pair an old stamp with a newer `Clean`.
    fn observed_clean(&self, dep: NodeId, rev: u64) -> bool {
        let node = self.nodes.get(dep.0);
        let state = node.state();
        (state.is_settled()
            || state == NodeState::Dirty && self.unchanged_stash(dep.0, node.generation()))
            && node.changed_at() <= rev
    }

    #[cfg(test)]
    pub(crate) fn arena<T: Value>(&self) -> &GenericArena<T, C> {
        let tag = self.inner.read().arena_tags.tag::<T>();
        tag.and_then(|tag| self.arenas.tagged_arena::<T>(tag))
            .expect("incr-core: arena lookup failed for T")
    }
}

// SAFETY: Runtime<Shared> is Send + Sync by composition (SegmentedNodes,
// RwLock, AtomicU64, SharedDepStack, RuntimeId all are). Runtime<Local>
// uses Cell/RefCell-backed cells through the Local strategy and is
// !Send + !Sync by auto-derive. We rely on auto traits here; no manual
// impls needed.

#[cfg(any(test, feature = "test-hooks"))]
impl<C: Cells> Runtime<C> {
    /// Invariant (I) of the shared-protocol plan, checked at quiescence:
    /// every installed dep S of a `Clean` node X is alive, has X in
    /// `dependents[S]`, and is itself `Clean`. Also checks that no freed
    /// slot still carries reverse edges.
    pub(crate) fn assert_invariant_i(&self) {
        let inner = self.inner.read();
        for x in 0..self.nodes.len() {
            let node = self.nodes.get(x);
            if node.state() != NodeState::Clean {
                continue;
            }
            node.for_each_dep(|s| {
                let dep = self.nodes.get(s.0);
                assert!(
                    dep.state() != NodeState::Dead && !inner.free_nodes.contains(&s.0),
                    "(I) violated: Clean node {x} has dead dep {}",
                    s.0
                );
                assert!(
                    inner.dependents[s.0 as usize].contains(NodeId(x)),
                    "(I) violated: Clean node {x} is missing from dependents[{}]",
                    s.0
                );
                assert!(
                    dep.state().is_settled(),
                    "(I) violated: Clean node {x} has dep {} {:?}",
                    s.0,
                    dep.state()
                );
            });
        }
        for slot in 0..self.nodes.len() as usize {
            let mut ids: Vec<NodeId> = inner.dependents[slot].iter().collect();
            let n = ids.len();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), n, "dependents[{slot}] holds a node twice");
            assert_eq!(
                C::u8_load_relaxed(self.nodes.get_with_flags(slot as u32).1) & HAS_CHILDREN != 0,
                inner.owned.contains_key(&(slot as u32)),
                "HAS_CHILDREN of node {slot} disagrees with the owned map"
            );
        }
        for slot in 0..self.nodes.len() {
            if inner.free_nodes.contains(&slot) {
                continue;
            }
            assert_eq!(
                C::u8_load_relaxed(self.nodes.get_with_flags(slot).1) & IN_GROUP != 0,
                inner.member_recs.contains_key(&slot),
                "IN_GROUP of node {slot} disagrees with the member records"
            );
        }
        for &slot in &inner.free_nodes {
            assert!(
                inner.dependents[slot as usize].is_empty(),
                "freed slot {slot} still has reverse edges {:?}",
                inner.dependents[slot as usize]
            );
        }
        for slot in 0..self.nodes.len() {
            assert_eq!(
                C::u8_load_relaxed(self.nodes.get_with_flags(slot).1) & HAS_INDEX != 0,
                inner.children.contains_key(&slot),
                "HAS_INDEX of node {slot} disagrees with the child indexes"
            );
        }
        for (&owner, owned) in &inner.owned {
            for c in owned {
                assert_eq!(
                    inner.owner_of.get(&c.0),
                    Some(&NodeId(owner)),
                    "node {owner} owns {} but owner_of disagrees",
                    c.0
                );
            }
        }
        assert_eq!(
            inner.owner_of.len(),
            inner.owned.values().map(Vec::len).sum::<usize>(),
            "owner_of and owned disagree"
        );
        for slot in 0..self.nodes.len() {
            if inner.free_nodes.contains(&slot) {
                continue;
            }
            assert_eq!(
                C::u8_load_relaxed(self.nodes.get_with_flags(slot).1) & OWNED != 0,
                inner.owner_of.contains_key(&slot),
                "OWNED of node {slot} disagrees with owner_of"
            );
        }
        for (&creator, index) in &inner.children {
            for child in index.with(|s| s.children()) {
                if self.nodes.get(child.slot()).generation() == child.generation() {
                    assert_eq!(
                        inner.owner_of.get(&child.slot()),
                        Some(&NodeId(creator)),
                        "the index of {creator} names {} which it does not own",
                        child.slot()
                    );
                }
            }
        }
    }

    /// Every installed dep of a node that is not `Dead` is alive and has
    /// the node in its dependents. Covers `Dirty`, `Failed` and recovered
    /// `New` nodes, where a missing edge would otherwise surface only
    /// after the node's next fast-path publish.
    pub(crate) fn assert_edges_complete(&self) {
        let inner = self.inner.read();
        for x in 0..self.nodes.len() {
            let node = self.nodes.get(x);
            if node.state() == NodeState::Dead {
                continue;
            }
            node.for_each_dep(|s| {
                assert!(
                    self.nodes.get(s.0).state() != NodeState::Dead
                        && !inner.free_nodes.contains(&s.0),
                    "node {x} ({:?}) has dead dep {}",
                    node.state(),
                    s.0
                );
                assert!(
                    inner.dependents[s.0 as usize].contains(NodeId(x)),
                    "node {x} ({:?}) is missing from dependents[{}]",
                    node.state(),
                    s.0
                );
            });
        }
    }
}

#[cfg(test)]
impl<C: Cells> Runtime<C> {
    #[cfg(not(loom))]
    fn computing_flags(&self) -> u16 {
        let (ctl, _) = (self.dep_stack.active_frame(self.runtime_id)).expect("a compute is active");
        // SAFETY: an active frame's control block is live while the frame
        // is on the stack (see `Frame::ctl`).
        unsafe { (*ctl).flags.get() }
    }

    #[cfg(not(loom))]
    /// Whether this runtime's active compute claimed its node New, so the
    /// node has landed nothing since it was made.
    pub(crate) fn computing_first(&self) -> bool {
        self.computing_flags() & cycle::RECOMPUTE == 0
    }

    #[cfg(not(loom))]
    pub(crate) fn computing_speculative(&self) -> bool {
        self.computing_flags() & cycle::SPECULATIVE != 0
    }

    #[cfg(not(loom))]
    /// Give `member` installed edges to `extra` on top of its own, with
    /// the reverse edges, as an episode leaves a cycle's members. It
    /// installs without the member's claim, so no other thread may run.
    pub(crate) fn add_edges_for_test<T, U>(&self, member: Incr<T>, extra: &[Incr<U>]) {
        let mut inner = self.inner.write();
        let id = NodeId(member.slot());
        let node = self.nodes.get(id.0);
        let mut deps = Vec::new();
        node.for_each_dep(|d| deps.push(d));
        for e in extra {
            deps.push(NodeId(e.slot()));
            inner.dependents[e.slot() as usize].push_unique(id);
        }
        if let Some(displaced) = node.install_deps(&deps) {
            drop(displaced);
        }
    }

    #[cfg(not(loom))]
    pub(crate) fn force_state_for_test<T>(&self, h: Incr<T>, to: NodeState) {
        state::store::<C>(self.nodes.get(h.slot()).state_cell(), to);
    }

    /// Moves a never-run query from New to Dirty, as no engine path does,
    /// so that a verify meets an empty slot behind `verified_at = 0`.
    #[cfg(not(loom))]
    pub(crate) fn force_dirty_for_test<T>(&self, h: Incr<T>) {
        let node = self.nodes.get(h.slot());
        assert_eq!(node.dep_count(), 0, "a forced node has no installed deps");
        assert_eq!(node.verified_at(), 0, "a forced node was never verified");
        state::try_transition::<C>(node.state_cell(), NodeState::New, NodeState::Dirty)
            .expect("a forced node is New");
    }

    /// Records `members` as one landed group, as an episode's step 2 does.
    pub(crate) fn force_group_for_test(&self, members: &[ErasedIncr]) {
        let mut inner = self.inner.write();
        let mut sorted: Vec<(u32, u32)> =
            members.iter().map(|m| (m.slot(), m.generation())).collect();
        sorted.sort_unstable();
        let group = Arc::new(GroupInfo {
            members: sorted,
            cycle: Cycle::new(members.to_vec()),
        });
        for m in members {
            inner.member_recs.insert(m.slot(), group.clone());
            let flags = self.nodes.get_with_flags(m.slot()).1;
            C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) | IN_GROUP);
        }
    }

    #[cfg(not(loom))]
    pub(crate) fn verified_at_of<T>(&self, h: Incr<T>) -> u64 {
        self.nodes.get(h.slot()).verified_at()
    }

    #[cfg(not(loom))]
    pub(crate) fn walk_pops(&self) -> u64 {
        self.walk_pops.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn has_reverse_edge<T, U>(&self, dep: Incr<T>, dependent: Incr<U>) -> bool {
        self.inner.read().dependents[dep.slot() as usize].contains(NodeId(dependent.slot()))
    }

    pub(crate) fn state_of<T>(&self, handle: Incr<T>) -> NodeState {
        self.nodes.get(handle.slot()).state()
    }

    pub(crate) fn assert_walks_complete(&self) {
        assert_eq!(
            C::walked(&self.walk, &self.revision),
            self.current_revision(),
            "the walked watermark lags the revision at quiescence"
        );
    }
}

#[path = "runtime_cycle.rs"]
mod cycle_impl;

#[path = "runtime_children.rs"]
mod children_impl;

#[path = "runtime_claim_set.rs"]
mod claim_set_impl;

#[path = "runtime_replay.rs"]
mod replay_impl;

#[path = "runtime_provenance.rs"]
mod provenance_impl;

#[path = "runtime_contain.rs"]
mod contain_impl;

#[path = "runtime_watch.rs"]
mod watch_impl;

#[path = "runtime_observe.rs"]
mod observe_impl;

#[path = "runtime_seed.rs"]
mod seed_impl;

#[cfg(all(test, not(loom)))]
#[path = "runtime_seed_tests.rs"]
mod seed_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_watch_tests.rs"]
mod watch_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_observer_read_tests.rs"]
mod observer_read_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_stabilize_tests.rs"]
mod stabilize_tests;

pub(crate) use children_impl::ChildLanding;
pub(crate) use observe_impl::ObserverEntry;
use observe_impl::Observers;

#[cfg(all(test, not(loom)))]
#[path = "runtime_children_tests.rs"]
mod children_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_replay_tests.rs"]
mod replay_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_replay_cycle_tests.rs"]
mod replay_cycle_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_replay_unwind_tests.rs"]
mod replay_unwind_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_replay_scale_tests.rs"]
mod replay_scale_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_replay_race_tests.rs"]
mod replay_race_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_reserve_tests.rs"]
mod reserve_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_replay_differential.rs"]
mod replay_differential;

use cycle_impl::{Attempt, GroupEnd, PopEnd};

#[cfg(all(test, not(loom)))]
#[path = "runtime_race_tests.rs"]
pub(crate) mod race_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_set_tests.rs"]
mod set_tests;

#[cfg(test)]
#[path = "runtime_cascade_tests.rs"]
mod cascade_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_walk_tests.rs"]
mod walk_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_cycle_walk_tests.rs"]
mod cycle_walk_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_cycle_tests.rs"]
mod cycle_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_cycle_rule_tests.rs"]
mod cycle_rule_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_wound_tests.rs"]
pub(crate) mod wound_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_value_code_tests.rs"]
pub(crate) mod value_code_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_drop_panic_tests.rs"]
mod drop_panic_tests;

#[cfg(all(test, not(loom)))]
#[path = "runtime_value_code_race_tests.rs"]
mod value_code_race_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_create_and_get_input() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(42_u64);
        assert_eq!(rt.get(a), 42);
    }

    #[test]
    fn shared_create_and_get_input() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(42_u64);
        assert_eq!(rt.get(a), 42);
    }

    #[test]
    fn local_simple_query() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
    }

    #[test]
    fn shared_simple_query() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
    }

    #[test]
    fn local_set_propagates() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
        rt.set(a, 15);
        assert_eq!(rt.get(b), 30);
    }

    #[test]
    fn shared_set_propagates() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(10_i64);
        let b = rt.create_query(move |rt| rt.get(a) * 2);
        assert_eq!(rt.get(b), 20);
        rt.set(a, 15);
        assert_eq!(rt.get(b), 30);
    }

    #[test]
    fn local_chain() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(5_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 1);
        let c = rt.create_query(move |rt| rt.get(b) * 2);
        assert_eq!(rt.get(c), 12);
        rt.set(a, 10);
        assert_eq!(rt.get(c), 22);
    }

    #[test]
    fn shared_chain() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(5_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 1);
        let c = rt.create_query(move |rt| rt.get(b) * 2);
        assert_eq!(rt.get(c), 12);
        rt.set(a, 10);
        assert_eq!(rt.get(c), 22);
    }

    #[test]
    fn local_diamond() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(a) + 100);
        let d = rt.create_query(move |rt| rt.get(b) + rt.get(c));
        assert_eq!(rt.get(d), 112);
        rt.set(a, 2);
        assert_eq!(rt.get(d), 114);
    }

    #[test]
    fn shared_diamond() {
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(a) + 100);
        let d = rt.create_query(move |rt| rt.get(b) + rt.get(c));
        assert_eq!(rt.get(d), 112);
        rt.set(a, 2);
        assert_eq!(rt.get(d), 114);
    }

    #[test]
    fn local_only_affected_recompute() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_input(2_i64);

        let ca = Arc::new(AtomicU32::new(0));
        let cb = Arc::new(AtomicU32::new(0));

        let ca_clone = ca.clone();
        let derived_a = rt.create_query(move |rt| {
            ca_clone.fetch_add(1, Ordering::Relaxed);
            rt.get(a) * 10
        });
        let cb_clone = cb.clone();
        let derived_b = rt.create_query(move |rt| {
            cb_clone.fetch_add(1, Ordering::Relaxed);
            rt.get(b) * 10
        });

        assert_eq!(rt.get(derived_a), 10);
        assert_eq!(rt.get(derived_b), 20);
        assert_eq!(ca.load(Ordering::Relaxed), 1);
        assert_eq!(cb.load(Ordering::Relaxed), 1);

        rt.set(a, 5);
        assert_eq!(rt.get(derived_a), 50);
        assert_eq!(rt.get(derived_b), 20);
        assert_eq!(ca.load(Ordering::Relaxed), 2);
        assert_eq!(cb.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn shared_early_cutoff_stops_propagation() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let rt: Runtime<Shared> = Runtime::new();
        let a = rt.create_input(50_i64);

        let c_count = Arc::new(AtomicU32::new(0));
        let cc = c_count.clone();
        let b = rt.create_query(move |rt| rt.get(a).min(100));
        let c = rt.create_query(move |rt| {
            cc.fetch_add(1, Ordering::Relaxed);
            rt.get(b) + 1
        });

        assert_eq!(rt.get(c), 51);
        assert_eq!(c_count.load(Ordering::Relaxed), 1);

        rt.set(a, 200);
        assert_eq!(rt.get(c), 101);
        assert_eq!(c_count.load(Ordering::Relaxed), 2);

        // a=300 → b still clamps to 100 → c skipped via early cutoff.
        rt.set(a, 300);
        assert_eq!(rt.get(c), 101);
        assert_eq!(c_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn local_query_with_more_than_seven_deps() {
        // Exercises the overflow path on NodeData::install_deps and
        // for_each_dep.
        let rt: Runtime<Local> = Runtime::new();
        let inputs: Vec<_> = (0..12_i64).map(|v| rt.create_input(v)).collect();
        let captured = inputs.clone();
        let sum = rt.create_query(move |rt| {
            let mut total = 0_i64;
            for i in &captured {
                total += rt.get(*i);
            }
            total
        });
        // 0+1+2+...+11 = 66
        assert_eq!(rt.get(sum), 66);
        // Mutate one input and verify it propagates.
        rt.set(inputs[5], 100);
        // 0+1+2+3+4+100+6+7+8+9+10+11 = 161
        assert_eq!(rt.get(sum), 161);
    }

    #[test]
    fn shared_query_with_more_than_seven_deps() {
        let rt: Runtime<Shared> = Runtime::new();
        let inputs: Vec<_> = (0..15_i64).map(|v| rt.create_input(v)).collect();
        let captured = inputs.clone();
        let sum = rt.create_query(move |rt| {
            let mut total = 0_i64;
            for i in &captured {
                total += rt.get(*i);
            }
            total
        });
        // sum 0..15 = 105
        assert_eq!(rt.get(sum), 105);
        rt.set(inputs[10], 1000);
        // 0+1+...+9 + 1000 + 11+12+13+14 = 45 + 1000 + 50 = 1095
        assert_eq!(rt.get(sum), 1095);
    }

    #[test]
    fn local_get_traced_records_recompute_events() {
        use crate::trace::TraceAction;
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_query(move |rt| rt.get(a) + 10);
        let c = rt.create_query(move |rt| rt.get(b) * 2);
        let _ = rt.get(c);

        // Set then traced read: every dirty node should appear in the trace.
        rt.set(a, 5);
        let (value, trace) = rt.get_traced(c);
        assert_eq!(value, 30); // (5 + 10) * 2
        assert_eq!(trace.target, NodeId(c.slot()));
        assert_eq!(trace.nodes_recomputed, 2); // b and c both recomputed
        assert_eq!(trace.nodes_cutoff, 0);
        // Verify the trace has Recomputed events with value_changed=true
        let recomputed_count = trace
            .node_traces
            .iter()
            .filter(|t| {
                matches!(
                    t.action,
                    TraceAction::Recomputed {
                        value_changed: true
                    }
                )
            })
            .count();
        assert_eq!(recomputed_count, 2);
    }

    #[test]
    fn local_get_traced_records_early_cutoff() {
        use crate::trace::TraceAction;
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(200_i64);
        let clamped = rt.create_query(move |rt| rt.get(input).min(100));
        let downstream = rt.create_query(move |rt| rt.get(clamped) + 1);
        let _ = rt.get(downstream);

        // Set input > 100 again; clamped still produces 100, so downstream
        // gets early-cutoff (Recomputed with value_changed=false on clamped,
        // VerifiedClean on downstream because its dep didn't change_at).
        rt.set(input, 300);
        let (value, trace) = rt.get_traced(downstream);
        assert_eq!(value, 101);

        // clamped should have a Recomputed event with value_changed=false
        // (the cutoff).
        let cutoffs = trace
            .node_traces
            .iter()
            .filter(|t| {
                matches!(
                    t.action,
                    TraceAction::Recomputed {
                        value_changed: false
                    }
                )
            })
            .count();
        assert!(
            cutoffs >= 1,
            "expected at least one cutoff event, got trace {:?}",
            trace.node_traces
        );
        assert!(trace.nodes_cutoff >= 1);
    }

    #[test]
    #[should_panic(expected = "dependency cycle")]
    fn local_direct_cycle_panics() {
        use std::sync::OnceLock;
        let rt: Runtime<Local> = Runtime::new();
        let own: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let captured = own.clone();
        let q = rt.create_query(move |rt| rt.get(*captured.get().unwrap()) + 1);
        own.set(q).unwrap();
        let _ = rt.get(q);
    }

    #[test]
    #[should_panic(expected = "dependency cycle")]
    fn shared_direct_cycle_panics() {
        use std::sync::OnceLock;
        let rt: Runtime<Shared> = Runtime::new();
        let own: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let captured = own.clone();
        let q = rt.create_query(move |rt| rt.get(*captured.get().unwrap()) + 1);
        own.set(q).unwrap();
        let _ = rt.get(q);
    }

    #[test]
    #[should_panic(expected = "dependency cycle")]
    fn local_transitive_cycle_panics() {
        use std::sync::OnceLock;
        let rt: Runtime<Local> = Runtime::new();
        let q1_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let q1_for_q2 = q1_cell.clone();
        let q2 = rt.create_query(move |rt| rt.get(*q1_for_q2.get().unwrap()) * 2);
        let q1 = rt.create_query(move |rt| rt.get(q2) + 1);
        q1_cell.set(q1).unwrap();
        let _ = rt.get(q1);
    }

    /// A cycle that only forms after a set() makes a query start reading
    /// a node that transitively reads it back. The cycle is reached
    /// mid-compute, so the frame-stack check fires.
    #[test]
    #[should_panic(expected = "dependency cycle")]
    fn local_cycle_formed_by_dynamic_deps_panics() {
        use std::sync::OnceLock;
        let rt: Runtime<Local> = Runtime::new();
        let s = rt.create_input(0_i64);
        let q1_cell: Arc<OnceLock<Incr<i64>>> = Arc::new(OnceLock::new());
        let q1_for_q2 = q1_cell.clone();
        let q2 = rt.create_query(move |rt| {
            if rt.get(s) == 1 {
                rt.get(*q1_for_q2.get().unwrap())
            } else {
                0
            }
        });
        let q1 = rt.create_query(move |rt| if rt.get(s) == 1 { rt.get(q2) } else { 0 });
        q1_cell.set(q1).unwrap();
        assert_eq!(rt.get(q1), 0);
        rt.set(s, 1);
        let _ = rt.get(q1);
    }

    /// Many threads race the FIRST get of the same query chain. Before
    /// the wait-on-Computing protocol, claim losers fell through to the
    /// arena read and hit an uninitialized slot.
    #[test]
    fn shared_first_get_claim_race_returns_correct_value() {
        use std::sync::Barrier;
        for _ in 0..50 {
            let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
            let a = rt.create_input(3_i64);
            let b = rt.create_query(move |rt| rt.get(a) * 7);
            let c = rt.create_query(move |rt| rt.get(b) + 1);
            let barrier = Arc::new(Barrier::new(8));
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let rt = Arc::clone(&rt);
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        rt.get(c)
                    })
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), 22);
            }
        }
    }

    /// Concurrent setters on distinct inputs must not lose revision
    /// increments or strand stale cutoffs: the sum must be exact after
    /// all writers join.
    #[test]
    fn shared_concurrent_setters_converge() {
        let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
        let inputs: Vec<_> = (0..8).map(|_| rt.create_input(0_i64)).collect();
        let captured = inputs.clone();
        let sum = rt.create_query(move |rt| captured.iter().map(|i| rt.get(*i)).sum::<i64>());
        assert_eq!(rt.get(sum), 0);

        let handles: Vec<_> = inputs
            .iter()
            .map(|&input| {
                let rt = Arc::clone(&rt);
                std::thread::spawn(move || {
                    for v in 1..=500_i64 {
                        rt.set(input, v);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(rt.get(sum), 500 * 8);
    }

    /// Readers hammer a query while a writer mutates its input. Every
    /// observed value must be a consistent function of SOME input the
    /// writer published, and reads must never go backwards once the
    /// writer is done.
    #[test]
    fn shared_reader_writer_hammer_observes_consistent_values() {
        let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
        let input = rt.create_input(0_i64);
        let doubled = rt.create_query(move |rt| rt.get(input) * 2);
        assert_eq!(rt.get(doubled), 0);

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let rt = Arc::clone(&rt);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let v = rt.get(doubled);
                        assert!(v % 2 == 0 && (0..=2000).contains(&v), "torn read: {}", v);
                    }
                })
            })
            .collect();

        for v in 1..=1000_i64 {
            rt.set(input, v);
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
        assert_eq!(rt.get(doubled), 2000);
    }

    #[test]
    fn observe_and_stabilize_fire_on_change_only() {
        use std::sync::Mutex;
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(1_i64);
        let doubled = rt.create_query(move |rt| rt.get(input) * 2);

        let seen: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_cb = Arc::clone(&seen);
        let id = rt.observe(doubled, move |v| seen_cb.lock().unwrap().push(*v));

        // Never-computed node fires on first stabilize.
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![2]);

        // No change, no fire.
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![2]);

        // Change fires once per stabilize, with the new value.
        rt.set(input, 5);
        rt.stabilize();
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![2, 10]);

        // Unobserve stops notifications.
        rt.unobserve(id);
        rt.set(input, 9);
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![2, 10]);
    }

    #[test]
    fn observe_respects_early_cutoff() {
        use std::sync::Mutex;
        let rt: Runtime<Shared> = Runtime::new();
        let input = rt.create_input(200_i64);
        let clamped = rt.create_query(move |rt| rt.get(input).min(100));
        let _ = rt.get(clamped);

        let seen: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_cb = Arc::clone(&seen);
        rt.observe(clamped, move |v| seen_cb.lock().unwrap().push(*v));

        // Input changes but the clamped value does not: cutoff, no fire.
        rt.set(input, 300);
        rt.stabilize();
        assert_eq!(seen.lock().unwrap().len(), 0);

        rt.set(input, 50);
        rt.stabilize();
        assert_eq!(*seen.lock().unwrap(), vec![50]);
    }

    #[test]
    fn delete_node_drops_its_observers() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(1_i64);
        let fired = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let fired_cb = Arc::clone(&fired);
        rt.observe(input, move |_| {
            fired_cb.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        rt.delete_node(input);
        // The observer is gone; stabilize must not touch the dead slot.
        rt.stabilize();
        assert_eq!(fired.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn delete_node_recycles_slot_and_rejects_stale_handles() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        let b = rt.create_input(2_i64);
        assert_eq!(rt.node_count(), 2);

        rt.delete_node(a);
        assert_eq!(rt.node_count(), 1);

        // The freed slot is recycled, under a new generation.
        let c = rt.create_input(30_i64);
        assert_eq!(c.slot(), a.slot());
        assert_ne!(c, a);
        assert_eq!(rt.node_count(), 2);
        assert_eq!(rt.get(c), 30);
        assert_eq!(rt.get(b), 2);

        let stale = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(a)));
        let msg = *stale.unwrap_err().downcast::<String>().unwrap();
        assert!(msg.contains("stale"), "got: {}", msg);
    }

    #[test]
    fn delete_query_then_input_unhooks_edges() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(5_i64);
        let q = rt.create_query(move |rt| rt.get(input) * 2);
        assert_eq!(rt.get(q), 10);

        rt.delete_node(q);
        // The input no longer has dependents, so it can go too.
        rt.delete_node(input);
        assert_eq!(rt.node_count(), 0);

        // set on the deleted input must fail, not corrupt a recycled slot.
        let fresh = rt.create_input("hello".to_string());
        assert_eq!(rt.get(fresh), "hello");
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.set(input, 9))).is_err()
        );
    }

    #[test]
    #[should_panic(expected = "still has dependents")]
    fn delete_node_with_dependents_panics() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(5_i64);
        let _q = rt.create_query(move |rt| rt.get(input) * 2);
        let _ = rt.get(_q);
        rt.delete_node(input);
    }

    struct RtPtr<C: Cells>(*const Runtime<C>);
    // SAFETY: test-only. Every runtime outlives the closures holding its
    // pointer, and a Local runtime's pointer never leaves its thread.
    unsafe impl<C: Cells> Send for RtPtr<C> {}
    unsafe impl<C: Cells> Sync for RtPtr<C> {}

    /// Runs runtime calls from its Drop, as a user capture might.
    struct TouchOnDrop<C: Cells> {
        rt: RtPtr<C>,
        probe: Incr<i64>,
        victim: Incr<i64>,
        dropped: Arc<std::sync::atomic::AtomicU32>,
    }

    impl<C: Cells> Drop for TouchOnDrop<C> {
        fn drop(&mut self) {
            // SAFETY: see RtPtr.
            let rt = unsafe { &*self.rt.0 };
            assert!(rt.is_alive(self.probe));
            rt.delete_node(self.victim);
            self.dropped
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn closures_drop_outside_the_lock<C: Cells>(rt: &Runtime<C>) {
        use std::sync::atomic::{AtomicU32, Ordering};
        let probe = rt.create_input(1_i64);
        let victims = [2, 3, 4].map(|v| rt.create_input(v as i64));
        let dropped = Arc::new(AtomicU32::new(0));
        let guard = |victim| TouchOnDrop {
            rt: RtPtr(rt as *const Runtime<C>),
            probe,
            victim,
            dropped: Arc::clone(&dropped),
        };

        let g = guard(victims[0]);
        let q = rt.create_query(move |rt| {
            let _keep = &g;
            rt.get(probe)
        });
        assert_eq!(rt.get(q), 1);
        let g = guard(victims[1]);
        rt.observe(q, move |_| {
            let _keep = &g;
        });
        let g = guard(victims[2]);
        let watcher = rt.observe(probe, move |_| {
            let _keep = &g;
        });

        rt.unobserve(watcher);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        rt.delete_node(q);
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
        assert!(victims.iter().all(|&v| !rt.is_alive(v)));
        assert!(rt.is_alive(probe));
    }

    #[test]
    fn local_closures_drop_outside_the_lock() {
        closures_drop_outside_the_lock(&Runtime::<Local>::new());
    }

    #[test]
    fn shared_closures_drop_outside_the_lock() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            closures_drop_outside_the_lock(&Runtime::<Shared>::new());
            tx.send(()).unwrap();
        });
        rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("a closure drop deadlocked or panicked under the runtime lock");
    }

    #[test]
    fn is_alive_tracks_delete_and_recycle() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        assert!(rt.is_alive(a));
        assert!(rt.is_alive(a.erase()));

        rt.delete_node(a);
        assert!(!rt.is_alive(a));

        let b = rt.create_input("recycled".to_string());
        assert_eq!(b.slot(), a.slot());
        assert!(!rt.is_alive(a), "a stale handle matches a recycled slot");
        assert!(rt.is_alive(b));
    }

    #[test]
    fn is_alive_is_false_for_a_foreign_handle() {
        let rt1: Runtime<Shared> = Runtime::new();
        let rt2: Runtime<Shared> = Runtime::new();
        let a = rt1.create_input(1_i64);
        let _same_slot = rt2.create_input(2_i64);
        assert!(!rt2.is_alive(a));
    }

    #[test]
    #[should_panic(expected = "stale")]
    fn delete_node_twice_panics_stale() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(1_i64);
        rt.delete_node(a);
        rt.delete_node(a);
    }

    #[test]
    #[should_panic(expected = "foreign runtime")]
    fn delete_node_with_foreign_handle_panics() {
        let rt1: Runtime<Local> = Runtime::new();
        let rt2: Runtime<Local> = Runtime::new();
        let a = rt1.create_input(1_i64);
        let _ = rt2.create_input(2_i64);
        rt2.delete_node(a);
    }

    #[test]
    #[should_panic(expected = "own compute")]
    fn delete_node_from_its_own_compute_panics() {
        let rt: Runtime<Local> = Runtime::new();
        let me: Arc<std::sync::OnceLock<Incr<i64>>> = Arc::new(std::sync::OnceLock::new());
        let me_in = Arc::clone(&me);
        let q = rt.create_query(move |rt| {
            rt.delete_node(*me_in.get().unwrap());
            0_i64
        });
        me.set(q).unwrap();
        let _ = rt.get(q);
    }

    #[test]
    fn delete_erased_reports_dependents_without_panicking() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(5_i64);
        let q = rt.create_query(move |rt| rt.get(input) * 2);
        assert_eq!(rt.get(q), 10);

        assert_eq!(
            rt.delete_erased(input.erase()),
            DeleteOutcome::HasDependents
        );
        assert!(rt.is_alive(input));
        assert_eq!(rt.get(q), 10);

        assert_eq!(rt.delete_erased(q.erase()), DeleteOutcome::Deleted);
        assert_eq!(rt.delete_erased(q.erase()), DeleteOutcome::AlreadyGone);
        assert_eq!(rt.delete_erased(input.erase()), DeleteOutcome::Deleted);
        assert_eq!(rt.node_count(), 0);
    }

    fn try_delete_removes_nodes_without_dependents<C: Cells>(rt: &Runtime<C>) {
        let input = rt.create_input(5_i64);
        let q = rt.create_query(move |rt| rt.get(input) * 2);
        assert_eq!(rt.get(q), 10);

        assert_eq!(rt.try_delete_node(q), Ok(()));
        assert!(!rt.is_alive(q));
        assert_eq!(rt.try_delete_node(input.erase()), Ok(()));
        assert_eq!(rt.node_count(), 0);
        assert_eq!(rt.try_delete_node(input), Err(DeleteError::NotAlive));
    }

    #[test]
    fn local_try_delete_removes_nodes_without_dependents() {
        try_delete_removes_nodes_without_dependents(&Runtime::<Local>::new());
    }

    #[test]
    fn shared_try_delete_removes_nodes_without_dependents() {
        try_delete_removes_nodes_without_dependents(&Runtime::<Shared>::new());
    }

    fn try_delete_with_dependents_keeps_the_node<C: Cells>(rt: &Runtime<C>) {
        let input = rt.create_input(5_i64);
        let _never_ran = rt.create_query(move |rt| rt.get(input) * 2);
        assert_eq!(
            rt.try_delete_node(input),
            Ok(()),
            "a query that never ran has no edge"
        );

        let input = rt.create_input(5_i64);
        let q2 = rt.create_query(move |rt| rt.get(input) * 2);
        assert_eq!(rt.get(q2), 10);
        assert_eq!(rt.try_delete_node(input), Err(DeleteError::HasDependents));
        assert!(rt.is_alive(input));
        rt.set(input, 6);
        assert_eq!(rt.get(q2), 12);
        rt.assert_invariant_i();
    }

    #[test]
    fn local_try_delete_with_dependents_keeps_the_node() {
        try_delete_with_dependents_keeps_the_node(&Runtime::<Local>::new());
    }

    #[test]
    fn shared_try_delete_with_dependents_keeps_the_node() {
        try_delete_with_dependents_keeps_the_node(&Runtime::<Shared>::new());
    }

    fn try_delete_of_a_dead_handle_is_not_alive<C: Cells>(rt: &Runtime<C>, other: &Runtime<C>) {
        let a = rt.create_input(1_i64);
        rt.delete_node(a);
        let b = rt.create_input(2_i64);
        assert_eq!(b.slot(), a.slot());
        assert_eq!(rt.try_delete_node(a), Err(DeleteError::NotAlive));
        assert!(rt.is_alive(b), "a stale handle deleted the recycled node");

        let foreign = other.create_input(3_i64);
        assert_eq!(foreign.slot(), b.slot());
        assert_eq!(rt.try_delete_node(foreign), Err(DeleteError::NotAlive));
        assert!(rt.is_alive(b) && other.is_alive(foreign));
    }

    #[test]
    fn local_try_delete_of_a_dead_handle_is_not_alive() {
        try_delete_of_a_dead_handle_is_not_alive(&Runtime::<Local>::new(), &Runtime::new());
    }

    #[test]
    fn shared_try_delete_of_a_dead_handle_is_not_alive() {
        try_delete_of_a_dead_handle_is_not_alive(&Runtime::<Shared>::new(), &Runtime::new());
    }

    #[test]
    fn deleted_slot_recycles_across_value_types() {
        let rt: Runtime<Local> = Runtime::new();
        let a = rt.create_input(7_i64);
        rt.delete_node(a);
        let s = rt.create_input("typed".to_string());
        assert_eq!(s.slot(), a.slot());
        assert_eq!(rt.get(s), "typed");
        rt.set(s, "retyped".to_string());
        assert_eq!(rt.get(s), "retyped");
    }

    #[test]
    fn shared_delete_and_recycle_under_reads() {
        let rt: Arc<Runtime<Shared>> = Arc::new(Runtime::new());
        let stable = rt.create_input(100_i64);
        let q = rt.create_query(move |rt| rt.get(stable) + 1);
        assert_eq!(rt.get(q), 101);

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..2)
            .map(|_| {
                let rt = Arc::clone(&rt);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        assert_eq!(rt.get(q), 101);
                    }
                })
            })
            .collect();

        // Churn unrelated nodes through delete/recreate while readers run.
        for i in 0..500_i64 {
            let tmp = rt.create_input(i);
            assert_eq!(rt.get(tmp), i);
            rt.delete_node(tmp);
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
        assert_eq!(rt.node_count(), 2);
    }

    #[test]
    #[should_panic(expected = "foreign runtime")]
    fn get_with_foreign_handle_panics() {
        let rt1: Runtime<Local> = Runtime::new();
        let rt2: Runtime<Local> = Runtime::new();
        let a = rt1.create_input(1_i64);
        let _ = rt2.get(a);
    }

    #[test]
    #[should_panic(expected = "foreign runtime")]
    fn set_with_foreign_handle_panics() {
        let rt1: Runtime<Shared> = Runtime::new();
        let rt2: Runtime<Shared> = Runtime::new();
        let a = rt1.create_input(1_i64);
        rt2.set(a, 2);
    }

    /// A panicking compute closure must mark the node Failed, propagate
    /// the panic, keep the runtime usable, and recover once an upstream
    /// input changes.
    #[test]
    fn local_compute_panic_marks_failed_then_recovers() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(0_i64);
        let q = rt.create_query(move |rt| {
            let v = rt.get(input);
            assert!(v != 0, "intentional test panic");
            v * 10
        });
        let other = rt.create_query(move |rt| rt.get(input) + 1);

        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q)));
        assert!(first.is_err());

        // Unrelated nodes still work; the dep stack was unwound cleanly.
        assert_eq!(rt.get(other), 1);

        // Reading the failed node again is a deliberate panic with a
        // recognizable message, not a deadlock.
        let again = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q)));
        let msg = *again.unwrap_err().downcast::<String>().unwrap();
        assert!(msg.contains("last compute panicked"), "got: {}", msg);

        // A dependency change flips Failed -> Dirty and the recompute
        // succeeds.
        rt.set(input, 7);
        assert_eq!(rt.get(q), 70);
        assert_eq!(rt.get(other), 8);
    }

    #[test]
    fn shared_compute_panic_marks_failed_then_recovers() {
        let rt: Runtime<Shared> = Runtime::new();
        let input = rt.create_input(0_i64);
        let q = rt.create_query(move |rt| {
            let v = rt.get(input);
            assert!(v != 0, "intentional test panic");
            v * 10
        });

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(q))).is_err());
        rt.set(input, 3);
        assert_eq!(rt.get(q), 30);
    }

    /// A panic mid-chain must fail both the panicking node and let its
    /// parent's frame unwind; recovery recomputes the whole chain.
    #[test]
    fn local_nested_compute_panic_unwinds_both_frames() {
        let rt: Runtime<Local> = Runtime::new();
        let input = rt.create_input(0_i64);
        let child = rt.create_query(move |rt| {
            let v = rt.get(input);
            assert!(v != 0, "intentional test panic");
            v + 1
        });
        let parent = rt.create_query(move |rt| rt.get(child) * 2);

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(parent))).is_err());
        assert!(!rt.dep_stack.current_frame_active(rt.runtime_id));

        rt.set(input, 5);
        assert_eq!(rt.get(parent), 12);
    }

    fn frame_active_inside_a_compute_only<C: Cells>() {
        let rt: Runtime<C> = Runtime::new();
        assert!(!rt.frame_active());
        let q = rt.create_query(|rt| rt.frame_active());
        assert!(rt.get(q));
        assert!(!rt.frame_active());
    }

    #[test]
    fn local_frame_active_inside_a_compute_only() {
        frame_active_inside_a_compute_only::<Local>();
    }

    #[test]
    fn shared_frame_active_inside_a_compute_only() {
        frame_active_inside_a_compute_only::<Shared>();
    }

    /// A query may read more than 255 inputs; the u8 dep-count cell uses
    /// an overflow marker and the DepList length is authoritative.
    #[test]
    fn local_query_with_300_deps_propagates() {
        let rt: Runtime<Local> = Runtime::new();
        let inputs: Vec<_> = (0..300_i64).map(|v| rt.create_input(v)).collect();
        let captured = inputs.clone();
        let sum = rt.create_query(move |rt| captured.iter().map(|i| rt.get(*i)).sum::<i64>());
        let expected: i64 = (0..300).sum();
        assert_eq!(rt.get(sum), expected);
        rt.set(inputs[123], 10_000);
        assert_eq!(rt.get(sum), expected - 123 + 10_000);
    }

    /// Stress test: many dynamic-dep transitions through the
    /// overflow path. Each iteration the dynamic query selects a
    /// different subset of inputs, forcing publish_deps to allocate
    /// a fresh overflow DepList and retire the old one through the
    /// haphazard global domain. Drop must complete cleanly with no
    /// UAF on the retired lists; miri / ASan would catch any leak.
    #[test]
    fn local_dynamic_overflow_deps_retirement() {
        use std::cell::Cell as StdCell;
        let rt: Runtime<Local> = Runtime::new();
        let switch = rt.create_input(0_u8);
        let inputs: Vec<_> = (0..16_i64).map(|v| rt.create_input(v)).collect();

        let captured = inputs.clone();
        let dynamic = rt.create_query(move |rt| -> i64 {
            let s = rt.get(switch) as usize;
            let start = s % 8;
            let mut total = 0;
            let extra = StdCell::new(s % 4);
            let end = (start + 8 + extra.get()).min(captured.len());
            for &node in &captured[start..end] {
                total += rt.get(node);
            }
            total
        });

        for s in 1..=50_u8 {
            rt.set(switch, s);
            let _ = rt.get(dynamic);
        }
        drop(rt);
    }

    const BIG: usize = 50_000;
    const SHIFT: usize = 10_000;

    /// `BIG + SHIFT` inputs and a query that reads the first `BIG` in
    /// order while `flip` is false, and the last `BIG` in reverse after.
    fn big_reordered_query(
        rt: &Runtime<Local>,
        panic_after: bool,
    ) -> (Incr<bool>, Vec<Incr<i64>>, Incr<i64>) {
        let flip = rt.create_input(false);
        let inputs: Vec<_> = (0..(BIG + SHIFT) as i64)
            .map(|v| rt.create_input(v))
            .collect();
        let captured = inputs.clone();
        let sum = rt.create_query(move |rt| {
            if !rt.get(flip) {
                return captured[..BIG].iter().map(|&i| rt.get(i)).sum::<i64>();
            }
            let total = captured[SHIFT..].iter().rev().map(|&i| rt.get(i)).sum();
            assert!(!panic_after, "panic after the reordered reads");
            total
        });
        (flip, inputs, sum)
    }

    fn installed_deps(rt: &Runtime<Local>, h: Incr<i64>) -> Vec<NodeId> {
        let mut deps = Vec::new();
        rt.nodes.get(h.slot()).for_each_dep(|d| deps.push(d));
        deps
    }

    /// A recompute whose 50k-entry dep list is shifted and reversed
    /// publishes in time linear in the list, installing exactly the new
    /// list and moving the reverse edges with it.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "a timing test; dep_set::tests cover the hashed diff under Miri"
    )]
    fn local_large_reordered_dep_list_publishes_new_edges() {
        let rt: Runtime<Local> = Runtime::new();
        let (flip, inputs, sum) = big_reordered_query(&rt, false);
        assert_eq!(rt.get(sum), (0..BIG as i64).sum());

        rt.set(flip, true);
        let start = std::time::Instant::now();
        assert_eq!(rt.get(sum), (SHIFT as i64..(BIG + SHIFT) as i64).sum());
        let took = start.elapsed();
        assert!(took < std::time::Duration::from_secs(5), "took {took:?}");

        let expected: Vec<NodeId> = std::iter::once(NodeId(flip.slot()))
            .chain(inputs[SHIFT..].iter().rev().map(|i| NodeId(i.slot())))
            .collect();
        assert_eq!(installed_deps(&rt, sum), expected);
        for (i, &input) in inputs.iter().enumerate() {
            assert_eq!(rt.has_reverse_edge(input, sum), i >= SHIFT, "input {i}");
        }
        rt.assert_invariant_i();
    }

    /// A panicking recompute over the same lists publishes the union of
    /// the old and recorded deps, each once, in linear time.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "a timing test; dep_set::tests cover the hashed diff under Miri"
    )]
    fn local_large_panicked_dep_list_publishes_the_union() {
        let rt: Runtime<Local> = Runtime::new();
        let (flip, inputs, sum) = big_reordered_query(&rt, true);
        assert_eq!(rt.get(sum), (0..BIG as i64).sum());

        rt.set(flip, true);
        let start = std::time::Instant::now();
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.get(sum)));
        let took = start.elapsed();
        assert!(read.is_err());
        assert!(took < std::time::Duration::from_secs(5), "took {took:?}");

        let expected: Vec<NodeId> = std::iter::once(NodeId(flip.slot()))
            .chain(inputs[..BIG].iter().map(|i| NodeId(i.slot())))
            .chain(inputs[BIG..].iter().rev().map(|i| NodeId(i.slot())))
            .collect();
        assert_eq!(installed_deps(&rt, sum), expected);
        for &input in &inputs {
            assert!(rt.has_reverse_edge(input, sum));
        }
        rt.assert_edges_complete();
    }
}
