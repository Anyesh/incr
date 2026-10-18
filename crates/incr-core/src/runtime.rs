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

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use crate::arena_registry::{ArenaRegistry, ArenaTags};
use crate::cells::Cells;
use crate::deferred::{Deferred, DeleteBag, NodeDrops};
use crate::dep_stack::DepStack;
use crate::dependent_list::DependentList;
use crate::generic_arena::GenericArena;
use crate::handle::{ErasedIncr, Incr, RuntimeId};
use crate::locks::Lock;
use crate::node::{NodeData, NodeId, RetiredDeps};
use crate::reclaim_probe::{self, RetireSite};
use crate::segmented_nodes::SegmentedNodes;
use crate::set_stack;
use crate::slot_vec::SlotVec;
use crate::state::{self, NodeState};
use crate::taint_memo;
use crate::value::Value;
use crate::value_code;
use crate::value_slot::{Compared, Displaced};

/// Compute closure: takes a borrow of the runtime, the node's slot,
/// whether this is a recompute (true) or first compute (false), and the
/// attempt's `wrote` cell, set once the slot holds the new value. Returns
/// `true` if the value actually changed (for early-cutoff propagation),
/// `false` if it was the same as before.
pub(crate) type ComputeFn<C> =
    Box<dyn Fn(&Runtime<C>, u32, bool, &Cell<u8>) -> bool + Send + Sync + 'static>;

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
    pub(crate) observers: HashMap<u64, ObserverEntry<C>>,
    pub(crate) next_observer_id: u64,
    /// Nodes an operator created for its own output and adopted: it sets
    /// them but never reads them, so no reverse edge records them.
    /// Sparse, because only operators like `group_by` own nodes. Kept in
    /// step with `owner_of` and unhooked by the child's delete under the
    /// same write hold as its generation bump, so an entry always names
    /// the node that was adopted.
    pub(crate) owned: HashMap<u32, Vec<NodeId>>,
    pub(crate) owner_of: HashMap<u32, NodeId>,
}

/// Type-erased observer runner: takes the runtime and the observer's
/// last-seen stamp, returns the new stamp if the callback fired.
type ObserverRunFn<C> = Arc<dyn Fn(&Runtime<C>, u64) -> Option<u64> + Send + Sync>;

/// One registered observer: which slot it watches, the changed_at stamp
/// it last fired for, and a type-erased runner that ensures the node is
/// clean and invokes the user callback if the value changed.
pub(crate) struct ObserverEntry<C: Cells> {
    pub(crate) slot: u32,
    pub(crate) last_seen: u64,
    pub(crate) run: ObserverRunFn<C>,
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
            observers: HashMap::new(),
            next_observer_id: 1,
            owned: HashMap::new(),
            owner_of: HashMap::new(),
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
    /// In `ensure_clean`, just before `compute_one` on a visited node.
    BeforeComputeOne(NodeId),
    /// In `compute_one`, after the state load and before the claim CAS.
    BeforeClaim(NodeId),
    /// In red/green verification, after the dep scan found no change and
    /// before the landing CAS.
    VerifyAfterScan(NodeId),
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
    /// In a strict delete of an owner, under the read hold, after the
    /// owner's `Dead` claim and before the claims of the nodes it owns.
    DeleteBeforeOwnedClaims(NodeId),
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

/// One attempt of a strict delete to claim its whole ownership tree under
/// a single read hold.
enum TreeClaim {
    /// The root and every owned node are `Dead` and claimed. `owned` are
    /// this delete's claims, owners before the nodes they own; `foreign`
    /// were claimed by another delete after this one checked them.
    Claimed {
        owned: Vec<ErasedIncr>,
        foreign: Vec<ErasedIncr>,
    },
    /// The root's state moved between its load and the claim.
    Retry,
    /// This owned node is claimed by another delete; wait it out first.
    WaitFor(ErasedIncr),
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
    fn finish_written(mut self) {
        self.held = false;
        let rev = self.rt.bump_revision();
        let walk_done = WalkDone { rt: self.rt, rev };
        self.rt.park(ParkPoint::SetAfterBump);
        self.node.settle_changed_at(rev);
        self.node.release_set_claim(self.base, rev);
        set_stack::pop(self.rt.runtime_id, self.id.0);
        self.rt.mark_dependents_dirty(self.id);
        self.rt.park(ParkPoint::SetAfterWalk);
        drop(walk_done);
        std::mem::forget(self);
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

/// How `compute_one` left the node it was handed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Computed {
    /// The node is as settled as the visit allows.
    Done,
    /// The slot's generation moved during the call: the node was deleted
    /// and its slot recycled, so the closure read under the hold belongs
    /// to a node that no longer exists. Any claim taken was released
    /// with the discard landing, and nothing ran against the occupant.
    Gone,
}

/// `Entry::taint` of a visit entry.
const VISIT: u32 = u32::MAX;

/// `Entry::taint` of a visit entry whose node was last loaded Computing,
/// so the visit takes it as Computing without loading it again. A stale
/// hint costs one backoff, after which the state is always reloaded;
/// every load skipped is a branch point fewer in the loom models.
const VISIT_COMPUTING: u32 = u32::MAX - 1;

/// `Entry::taint` of a pending entry no skipped dep has reached.
const NO_TAINT: u32 = u32::MAX - 2;

/// One entry of `ensure_clean`'s walk. A visit (`taint >= VISIT_COMPUTING`)
/// expands `x`'s non-Clean deps; the pending entry is pushed on the first
/// visit and runs `x`'s compute once they are done.
#[derive(Copy, Clone)]
struct Entry {
    id: NodeId,
    /// `VISIT` or `VISIT_COMPUTING`, or on a pending entry the lowest
    /// frame depth among the Computing deps this thread holds that were
    /// skipped under it, or `NO_TAINT`. A tainted node is reachable only through edges that
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
struct Walk {
    work: Vec<Entry>,
    first_visits: u32,
    /// Exact mode: every node with a `(x, true)` entry on `work`, kept
    /// from the switch on so a popped `(x, false)` of one of them is
    /// skipped instead of expanded again. `None` below the switch, which
    /// keeps the walk free of set operations for small graphs.
    visited: Option<HashSet<NodeId>>,
}

impl Walk {
    #[inline]
    fn new(root: NodeId, root_state: NodeState) -> Self {
        Self {
            work: vec![Entry {
                id: root,
                taint: Self::visit_kind(root_state),
            }],
            first_visits: 0,
            visited: None,
        }
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
    fn visit(&mut self, id: NodeId) {
        self.push(Entry {
            id,
            taint: NO_TAINT,
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
        const KIND: [u32; 7] = {
            let mut kind = [VISIT; 7];
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
/// Under `Shared`, displaced values are reclaimed through hazard
/// pointers, so a value's `Drop` may run inside a later, unrelated
/// retire on any thread. When such a `Drop` panics, the rest of that
/// reclaim batch is leaked, never dropped and never freed; the runtime
/// stays usable.
pub struct Runtime<C: Cells> {
    pub(crate) nodes: SegmentedNodes<C>,
    /// Read with no lock; written only under `inner.write()`, through
    /// `Inner::arena_tags`.
    pub(crate) arenas: ArenaRegistry<C>,
    pub(crate) inner: <C as Cells>::Lock<Inner<C>>,
    pub(crate) revision: <C as Cells>::U64,
    pub(crate) walk: <C as Cells>::Watermark,
    pub(crate) dep_stack: <C as Cells>::DepStack,
    pub(crate) runtime_id: RuntimeId,
    /// `1` when `get_traced` is actively recording. Checked on every
    /// `compute_one` via a Relaxed load (~1 ns when disarmed) so the
    /// non-tracing hot path pays no measurable cost.
    pub(crate) tracing_armed: <C as Cells>::U8,
    #[cfg(test)]
    pub(crate) park_hook: std::sync::OnceLock<ParkHook>,
    #[cfg(test)]
    pub(crate) bag_hook: std::sync::OnceLock<BagHook>,
    #[cfg(test)]
    first_visits: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    expansions: std::sync::atomic::AtomicU64,
}

impl<C: Cells> Default for Runtime<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cells> Runtime<C> {
    pub fn new() -> Self {
        Self {
            nodes: SegmentedNodes::new(),
            arenas: ArenaRegistry::new(),
            inner: <<C as Cells>::Lock<Inner<C>> as Lock<Inner<C>>>::new(Inner::new()),
            revision: C::new_u64(1),
            walk: C::new_watermark(1),
            dep_stack: <C::DepStack as DepStack>::new(),
            runtime_id: RuntimeId::allocate(),
            tracing_armed: C::new_u8(0),
            #[cfg(test)]
            park_hook: std::sync::OnceLock::new(),
            #[cfg(test)]
            bag_hook: std::sync::OnceLock::new(),
            #[cfg(test)]
            first_visits: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            expansions: std::sync::atomic::AtomicU64::new(0),
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
        C::u64_fetch_add(&self.revision, 1) + 1
    }

    /// Create an input node with an initial value.
    pub fn create_input<T: Value>(&self, value: T) -> Incr<T> {
        assert!(
            !self.dep_stack.current_frame_active(),
            "create_input called during compute; not permitted",
        );
        self.create_input_unchecked(value)
    }

    /// Internal: create an input without the dep-stack-empty check.
    /// Used by operators like `group_by` that need to allocate
    /// sub-collection version nodes lazily from inside a compute closure.
    /// The caller is responsible for ensuring the new node is not a dep
    /// of the currently-computing node (i.e., the new node is downstream
    /// of the operator, not upstream).
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
        assert!(
            !self.dep_stack.current_frame_active(),
            "create_query called during compute; not permitted",
        );
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

            // Compute closure: invokes f, writes value, returns whether
            // the value changed (for early cutoff). `is_recompute=false`
            // means there's no prior value, so we always treat it as
            // changed; `is_recompute=true` compares against the stored
            // value via T's PartialEq.
            let arena_inner = arena;
            let compute: ComputeFn<C> = Box::new(
                move |rt: &Runtime<C>, slot: u32, is_recompute: bool, wrote: &Cell<u8>| -> bool {
                    let new_value = f(rt);
                    let node = rt.nodes.get(slot);
                    let changed = if is_recompute {
                        // Compare-and-publish in one slot session; the
                        // false return is the early cutoff.
                        arena_inner.write_if_changed(node.arena_slot(), new_value, wrote)
                    } else {
                        wrote.set(1);
                        drop(arena_inner.write(node.arena_slot(), new_value));
                        true
                    };
                    reclaim_probe::reached(RetireSite::ComputeWrite);
                    changed
                },
            );

            inner.compute_fns[slot as usize] = Some(compute);
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
        panic!(
            "incr-core: Incr<{}> handle from a foreign runtime (handle was minted by \
             runtime {}, used on runtime {})",
            std::any::type_name::<T>(),
            handle.runtime_id().get(),
            self.runtime_id.get(),
        );
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

    /// Register a callback fired by [`Self::stabilize`] whenever the
    /// node's value has changed since the last firing. Observation does
    /// not change the engine's pull model: nothing runs until someone
    /// calls `stabilize` (or `get`); the observer registry is a batch
    /// poll with last-seen bookkeeping, the same shape as Jane Street
    /// Incremental's observers.
    ///
    /// A node that has never computed fires on the first stabilize. A
    /// stabilize racing a concurrent `set` may fire twice for one
    /// change; it can never miss one.
    pub fn observe<T: Value, F>(&self, handle: Incr<T>, callback: F) -> ObserverId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        value_code::check();
        self.check_handle(handle);
        let slot = handle.slot();
        let node = self.nodes.get(slot);
        // Fire-baseline: if the node already has a clean value, only
        // future changes fire; a never-computed node starts at 0 so the
        // first stabilize computes and fires.
        let last_seen = match state::load::<C>(node.state_cell()) {
            NodeState::New => 0,
            _ => node.changed_at().min(self.current_revision()),
        };

        let run = Arc::new(move |rt: &Runtime<C>, last_seen: u64| -> Option<u64> {
            if rt.ensure_clean(NodeId(handle.slot()), handle.generation()) == Computed::Gone {
                rt.stale_handle(handle);
            }
            let node = rt.nodes.get(handle.slot());
            // A pending set() marker reads as u64::MAX; clamp to the
            // current revision so last_seen stays passable and the
            // settled stamp can still exceed it (duplicate, not lost).
            let stamp = node.changed_at().min(rt.current_revision());
            if stamp > last_seen {
                let value = rt.get(handle);
                callback(&value);
                Some(stamp)
            } else {
                None
            }
        });

        let mut inner = self.inner.write();
        let id = inner.next_observer_id;
        inner.next_observer_id += 1;
        inner.observers.insert(
            id,
            ObserverEntry {
                slot,
                last_seen,
                run,
            },
        );
        ObserverId(id)
    }

    /// Remove an observer. Safe to call with an already-removed id.
    pub fn unobserve(&self, id: ObserverId) {
        value_code::check();
        // Bound to a local so the entry, whose callback is user code,
        // drops after the write guard does.
        let removed = self.inner.write().observers.remove(&id.0);
        drop(removed);
    }

    /// Bring every observed node up to date and fire callbacks for the
    /// ones whose values changed since they last fired. Callbacks run
    /// with no engine lock held; they may read from the runtime but must
    /// not call `stabilize` reentrantly from another thread if duplicate
    /// notifications matter.
    pub fn stabilize(&self) {
        value_code::check();
        let entries: Vec<(u64, u64, ObserverRunFn<C>)> = {
            let inner = self.inner.read();
            inner
                .observers
                .iter()
                .map(|(&id, e)| (id, e.last_seen, Arc::clone(&e.run)))
                .collect()
        };
        for (id, last_seen, run) in entries {
            if let Some(new_stamp) = run(self, last_seen) {
                let mut inner = self.inner.write();
                if let Some(e) = inner.observers.get_mut(&id) {
                    e.last_seen = e.last_seen.max(new_stamp);
                }
            }
        }
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
        match self.delete_strict(root) {
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
        self.delete_strict(root).map_err(|refusal| match refusal {
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

    fn assert_not_own_compute(&self, id: NodeId) {
        assert!(
            !self.dep_stack.is_computing(id),
            "incr-core: delete_node called from inside node {}'s own compute",
            id.0,
        );
    }

    /// The strict delete behind `delete_node` and `try_delete_node`:
    /// `root` and everything it owns, transitively, or nothing.
    ///
    /// The whole tree is checked for dependents and claimed `Dead` under
    /// one read hold, the root first. Owned nodes are inputs (see
    /// [`Runtime::adopt`]), whose state only moves from Clean to `Dead`,
    /// so once the root's claim succeeds no owned claim can fail, except
    /// to another delete claiming that node concurrently. Such a node is
    /// waited out before the root is released, so the root's owned list
    /// is empty by then. A set holding a claimed node is waited out after
    /// the hold, before anything is retired.
    fn delete_strict(&self, root: ErasedIncr) -> Result<(), StrictRefusal> {
        debug_assert_eq!(root.runtime_id(), self.runtime_id);
        value_code::check();
        let id = NodeId(root.slot());
        let node = self.nodes.get(root.slot());
        if node.generation() != root.generation() {
            return Err(StrictRefusal::Gone);
        }
        self.assert_not_own_compute(id);
        self.refuse_delete_under_own_set(id);
        self.park(ParkPoint::DeleteAfterHandleCheck(id));

        let mut spins: u32 = 0;
        let (owned, foreign) = loop {
            let cur = state::load::<C>(node.state_cell());
            if node.generation() != root.generation() {
                return Err(StrictRefusal::Gone);
            }
            match cur {
                NodeState::Computing | NodeState::ComputingDirty => {
                    self.refuse_wait_on_computing(id);
                    spins += 1;
                    C::backoff(spins);
                    continue;
                }
                NodeState::Dead => {
                    self.wait_out_claim(root, &mut spins);
                    return Err(StrictRefusal::Gone);
                }
                _ => {}
            }
            match self.claim_tree(root, cur)? {
                TreeClaim::Claimed { owned, foreign } => break (owned, foreign),
                TreeClaim::Retry => {}
                TreeClaim::WaitFor(child) => self.wait_out_claim(child, &mut spins),
            }
        };
        self.park(ParkPoint::DeleteAfterCheck(id));
        for &member in std::iter::once(&root).chain(&owned) {
            self.wait_out_set(member, &mut spins);
        }
        for child in foreign {
            self.wait_out_claim(child, &mut spins);
        }
        self.release_tree(owned, root).drain(self);
        Ok(())
    }

    /// Unhooks and recycles an owned tree this thread claimed `Dead`,
    /// deepest members first and the root last, under one write hold.
    /// So every member's generation is bumped before any closure,
    /// observer or value `Drop` runs at the bag's drain: a `Drop` that
    /// deletes a member then panics as stale instead of waiting on this
    /// thread's own claim (bug h).
    fn release_tree(&self, owned: Vec<ErasedIncr>, root: ErasedIncr) -> DeleteBag<'_, C> {
        if owned.is_empty() {
            return self.release_claimed(root);
        }
        let mut retired = owned
            .into_iter()
            .rev()
            .chain(std::iter::once(root))
            .map(|member| self.retire_claimed(member))
            .collect::<Vec<Retired<C>>>()
            .into_iter();
        let mut inner = self.inner.write();
        let first = retired
            .next()
            .expect("incr-core: a batch release has a member");
        let (drops, dep_list) = self.unhook_retired(&mut inner, first);
        let mut bag = DeleteBag::new(drops, dep_list);
        bag.reserve_batch(retired.len());
        for member in retired {
            let (drops, dep_list) = self.unhook_retired(&mut inner, member);
            bag.push_node(drops, dep_list);
        }
        bag
    }

    /// One claim attempt of [`Runtime::delete_strict`], with the root's
    /// state last loaded as `cur`.
    fn claim_tree(&self, root: ErasedIncr, cur: NodeState) -> Result<TreeClaim, StrictRefusal> {
        let inner = self.inner.read();
        // Rechecked under the hold: a delete and recycle since the load of
        // `cur` would otherwise let the CAS claim the slot's next occupant.
        if self.nodes.get(root.slot()).generation() != root.generation() {
            return Err(StrictRefusal::Gone);
        }
        if !inner.dependents[root.slot() as usize].is_empty() {
            return Err(StrictRefusal::HasDependents(root.slot()));
        }
        let mut owned = Vec::new();
        // The walk's stack starts from the root's children, so a root that
        // owns nothing allocates nothing.
        if let Some(children) = inner.owned.get(&root.slot()) {
            let mut stack: Vec<u32> = children.iter().map(|c| c.0).collect();
            while let Some(slot) = stack.pop() {
                if !inner.dependents[slot as usize].is_empty() {
                    return Err(StrictRefusal::HasDependents(slot));
                }
                self.refuse_delete_under_own_set(NodeId(slot));
                let child = self.erased_at(slot);
                if self.nodes.get(slot).state() == NodeState::Dead {
                    return Ok(TreeClaim::WaitFor(child));
                }
                owned.push(child);
                if let Some(children) = inner.owned.get(&slot) {
                    stack.extend(children.iter().map(|c| c.0));
                }
            }
        }
        let root_node = self.nodes.get(root.slot());
        if state::try_transition::<C>(root_node.state_cell(), cur, NodeState::Dead).is_err() {
            return Ok(TreeClaim::Retry);
        }
        self.park(ParkPoint::DeleteBeforeOwnedClaims(NodeId(root.slot())));
        let mut claimed = Vec::with_capacity(owned.len());
        let mut foreign = Vec::new();
        for child in owned {
            // Owned nodes are inputs, which are Clean until a delete
            // claims them; a set's claim is not on `state`.
            let cell = self.nodes.get(child.slot()).state_cell();
            match state::try_transition::<C>(cell, NodeState::Clean, NodeState::Dead) {
                Ok(()) => claimed.push(child),
                Err(NodeState::Dead) => foreign.push(child),
                Err(other) => self.input_state_broken(NodeId(child.slot()), other),
            }
        }
        Ok(TreeClaim::Claimed {
            owned: claimed,
            foreign,
        })
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
            *spins += 1;
            C::backoff(*spins);
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
            *spins += 1;
            C::backoff(*spins);
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

    /// Delete `root` together with every node that depends on it,
    /// transitively, and every node any of them owns (such as the groups
    /// of a `group_by` router). Returns how many nodes this call deleted;
    /// a root that is already gone counts as done and returns 0.
    ///
    /// Dependents are found through the reverse edges a compute records,
    /// so a query that was created but has never run is not reached: it
    /// survives and panics with a stale-handle message on its first
    /// compute. Concurrent cascades and plain deletes over the same nodes
    /// are safe, and a node deleted by one of them is not counted here.
    /// A cascade does not terminate while other threads keep attaching
    /// new dependents to its nodes faster than it removes them.
    ///
    /// Panics if called from inside a compute closure: the computing
    /// node may depend on `root`, and the cascade would wait on it
    /// forever.
    pub fn delete_with_dependents(&self, root: impl Into<ErasedIncr>) -> usize {
        value_code::check();
        let root = root.into();
        assert!(
            !self.dep_stack.current_frame_active(),
            "incr-core: delete_with_dependents called during compute; not permitted",
        );
        if root.runtime_id() != self.runtime_id {
            panic!(
                "incr-core: delete_with_dependents with a handle from a foreign runtime \
                 (handle was minted by runtime {}, used on runtime {})",
                root.runtime_id().get(),
                self.runtime_id.get(),
            );
        }

        let mut deleted = 0;
        let mut stack = vec![root];
        while let Some(&top) = stack.last() {
            // Captured under the same hold that yields the slot, so the
            // generation is the linked node's (see delete_erased).
            let next = {
                let inner = self.inner.read();
                if self.nodes.get(top.slot()).generation() != top.generation() {
                    stack.pop();
                    continue;
                }
                let slot = top.slot() as usize;
                // The last dependent, because a delete removes it from this
                // list by scanning from the back, so tearing down a fan-in
                // of n stays linear in n.
                inner.dependents[slot]
                    .last()
                    .or_else(|| {
                        inner
                            .owned
                            .get(&top.slot())
                            .and_then(|o| o.first().copied())
                    })
                    .map(|n| self.erased_at(n.0))
            };
            if let Some(child) = next {
                stack.push(child);
                continue;
            }
            match self.delete_erased(top) {
                DeleteOutcome::Deleted => {
                    deleted += 1;
                    stack.pop();
                }
                DeleteOutcome::AlreadyGone => {
                    stack.pop();
                }
                // A dependent attached after the hint; walk to it next.
                DeleteOutcome::HasDependents => {}
            }
        }
        deleted
    }

    /// Record `child` as owned by `owner`: it is deleted with the owner,
    /// by strict delete and by the cascade. Both must be live, and the
    /// owner must not depend on the child, or the cascade would walk the
    /// cycle forever. The child must be an input, because a strict delete
    /// claims owned nodes after their owner and relies on an input's
    /// state never moving except to `Dead`.
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
        assert!(
            inner.compute_fns[child.slot() as usize].is_none(),
            "incr-core: adopt of node {}, which is not an input",
            child.slot()
        );
        let previous = inner.owner_of.insert(child.slot(), NodeId(owner.slot()));
        assert!(
            previous.is_none(),
            "incr-core: node {} adopted twice",
            child.slot()
        );
        inner
            .owned
            .entry(owner.slot())
            .or_default()
            .push(NodeId(child.slot()));
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
        self.assert_not_own_compute(id);
        self.refuse_delete_under_own_set(id);

        let mut spins: u32 = 0;
        loop {
            let cur = state::load::<C>(node.state_cell());
            if node.generation() != handle.generation() {
                return DeleteOutcome::AlreadyGone;
            }
            match cur {
                NodeState::Computing | NodeState::ComputingDirty => {
                    self.refuse_wait_on_computing(id);
                    spins += 1;
                    C::backoff(spins);
                }
                NodeState::Dead => {
                    // Another delete claimed the node and will bump the
                    // generation under its next write lock. Returning now
                    // would let a cascade re-find this slot in its
                    // parent's dependents and spin on it instead.
                    self.wait_out_claim(handle, &mut spins);
                    return DeleteOutcome::AlreadyGone;
                }
                _ => {
                    // The check and the claim share one read hold so no
                    // publish (which pushes reverse edges under the write
                    // lock) can land between them; clearing dependents
                    // below would otherwise drop its edge silently. The
                    // generation is rechecked under the hold because a
                    // delete and recycle between the load of `cur` and
                    // the CAS would otherwise let the CAS claim the
                    // slot's next occupant.
                    let claimed = {
                        let inner = self.inner.read();
                        if node.generation() != handle.generation() {
                            return DeleteOutcome::AlreadyGone;
                        }
                        if !inner.dependents[slot as usize].is_empty()
                            || inner.owned.contains_key(&slot)
                        {
                            return DeleteOutcome::HasDependents;
                        }
                        state::try_transition::<C>(node.state_cell(), cur, NodeState::Dead).is_ok()
                    };
                    if claimed {
                        self.park(ParkPoint::DeleteAfterCheck(id));
                        break;
                    }
                }
            }
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
        let node = self.nodes.get(slot);
        debug_assert!(
            !inner.owned.contains_key(&slot),
            "incr-core: released node {} still owns nodes",
            slot,
        );
        for d in &retired.old_deps {
            inner.dependents[d.0 as usize].remove(id);
        }
        inner.dependents[slot as usize].clear();
        if let Some(owner) = inner.owner_of.remove(&slot) {
            let siblings = inner
                .owned
                .get_mut(&owner.0)
                .expect("incr-core: owner_of names an owner with no owned list");
            siblings.retain(|&c| c != id);
            if siblings.is_empty() {
                inner.owned.remove(&owner.0);
            }
        }
        node.bump_generation();
        let compute = inner.compute_fns[slot as usize].take();
        let observers: Vec<ObserverEntry<C>> = inner
            .observers
            .extract_if(|_, e| e.slot == slot)
            .map(|(_, e)| e)
            .collect();
        inner.labels.remove(&slot);
        let arena = self.arenas.erased(retired.type_tag);
        inner.free_nodes.push(slot);
        let drops = NodeDrops {
            compute,
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
        self.check_runtime(handle);
        let slot = handle.slot();
        let node = self.nodes.get(slot);
        self.park(ParkPoint::GetAfterHandleCheck(NodeId(slot)));

        let observed = state::load::<C>(node.state_cell());
        if observed == NodeState::Clean {
            let (type_tag, arena_slot) = self.fields_under(handle, node);
            self.dep_stack.record_dep(NodeId(slot), handle.generation());
            return self.read_slot(handle, node, type_tag, arena_slot);
        }
        self.get_slow(handle, observed)
    }

    /// The non-Clean arm of [`Self::get`]: walk, then read. `observed`
    /// is the state `get` loaded.
    #[inline(never)]
    fn get_slow<T: Value>(&self, handle: Incr<T>, observed: NodeState) -> T {
        let slot = handle.slot();
        let node = self.nodes.get(slot);
        if node.generation() != handle.generation() {
            self.stale_handle(handle);
        }
        self.dep_stack.record_dep(NodeId(slot), handle.generation());

        let word = value_code::thread_word();
        if word & value_code::COUNTER_MASK != 0 {
            return self.value_code_read(handle, node);
        }
        // `get` found the node not Clean, so the walk starts at once.
        if self.clean_walk(NodeId(slot), handle.generation(), word, observed) == Computed::Gone {
            self.stale_handle(handle);
        }
        let (type_tag, arena_slot) = self.fields_under(handle, node);
        self.read_slot(handle, node, type_tag, arena_slot)
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
        match state::load::<C>(node.state_cell()) {
            NodeState::Clean => {
                let (type_tag, arena_slot) = self.fields_under(handle, node);
                self.read_slot(handle, node, type_tag, arena_slot)
            }
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
        self.park(ParkPoint::GetBeforeRead(NodeId(handle.slot())));
        let arena = self.arenas.tagged_arena::<T>(type_tag);
        let value = value_code::run(|| arena.and_then(|arena| arena.try_read(arena_slot)));
        C::fence_acquire();
        if node.generation() == handle.generation() {
            if let Some(value) = value {
                return value;
            }
        }
        self.read_failed(handle)
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
        let displaced = self.write_claimed(claim, arena, arena_slot, value);
        // The old value's user `Drop`, or the reclaim its retire may run,
        // happens with no claim held and none of the locks this set took.
        drop(displaced);
        reclaim_probe::reached(RetireSite::SetRetire);
    }

    /// Take `set`'s claim on `handle`'s input, with the arena and slot
    /// its value lives in.
    fn claim_input<T: Value>(
        &self,
        handle: Incr<T>,
    ) -> (SetClaim<'_, C>, &GenericArena<T, C>, u32) {
        let slot = handle.slot();
        let id = NodeId(slot);
        let node = self.nodes.get(slot);
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
            spins += 1;
            C::backoff(spins);
        }
    }

    /// `set` from its claim to its walk. Returns what the write
    /// displaced, if it wrote.
    fn write_claimed<T: Value>(
        &self,
        claim: SetClaim<'_, C>,
        arena: &GenericArena<T, C>,
        arena_slot: u32,
        value: T,
    ) -> Option<Displaced<T, C>> {
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
        // 4. settle changed_at to the concrete revision;
        // 5. claim release, then the dirty walk, whose Release stores
        //    publish all of the above to claimants that Acquire the
        //    Dirty state.
        match arena.compare_and_write(arena_slot, value, || node.mark_changed_pending()) {
            Compared::Equal(rejected) => {
                // The set that wrote this value released its claim before
                // its walk, and this call must not return before that
                // change is visible to its caller's dependents.
                let changed_at = node.changed_at();
                drop(claim);
                self.wait_walked(changed_at);
                drop(rejected);
                None
            }
            Compared::Replaced(displaced) => {
                claim.finish_written();
                displaced
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
    ///
    /// Known limitation: a cyclic graph whose cycle spans two threads'
    /// in-flight computes spins forever instead of panicking; cross-thread
    /// cycle detection needs a waits-for graph and is not implemented.
    #[inline]
    fn wait_on_claim(&self, id: NodeId, spins: u32) {
        if self.dep_stack.is_computing(id) {
            panic!(
                "incr-core: dependency cycle detected: node {} was read while \
                 its own compute was in progress",
                id.0,
            );
        }
        self.refuse_wait_on_computing(id);
        C::backoff(spins);
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

    #[cold]
    #[inline(never)]
    fn input_state_broken(&self, id: NodeId, state: NodeState) -> ! {
        panic!(
            "incr-core: internal invariant violated: input {} is {:?}; an input is Clean or Dead",
            id.0, state,
        );
    }
    /// BFS forward walk from `start`'s dependents, marking each Clean
    /// node as Dirty. Every transition is a CAS so the walk can never
    /// clobber a `Computing` claim it raced with.
    fn mark_dependents_dirty(&self, start: NodeId) {
        let mut queue: Vec<NodeId> = {
            let inner = self.inner.read();
            inner.dependents[start.0 as usize].iter().collect()
        };

        while let Some(id) = queue.pop() {
            let node = self.nodes.get(id.0);
            loop {
                let cur = state::load::<C>(node.state_cell());
                match cur {
                    NodeState::Clean | NodeState::Failed => {
                        if state::try_transition::<C>(node.state_cell(), cur, NodeState::Dirty)
                            .is_ok()
                        {
                            let inner = self.inner.read();
                            queue.extend(inner.dependents[id.0 as usize].iter());
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
                            queue.extend(inner.dependents[id.0 as usize].iter());
                            break;
                        }
                    }
                    NodeState::New => {
                        // A node that recovered from a stale dep lands New
                        // but keeps its old reverse edges, so a Failed
                        // dependent can only be reached through it.
                        // Never-run queries have no dependents. The visit
                        // is an RMW for the reason given on the Dirty arm.
                        if state::try_transition::<C>(node.state_cell(), cur, cur).is_ok() {
                            let inner = self.inner.read();
                            queue.extend(inner.dependents[id.0 as usize].iter());
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
        self.clean_walk(id, gen, word, observed)
    }

    /// [`Self::ensure_clean`] past its fast path, for a caller that took
    /// `word` from `value_code::checked_word` and loaded `id` as
    /// `observed`, not Clean. A root that went Clean since is left at its
    /// first pop, or after one backoff if it was loaded Computing.
    ///
    /// A node found Computing is waited on here once before its pending
    /// entry's compute loads it again, so that no Computing state is
    /// loaded twice before the first backoff.
    fn clean_walk(&self, id: NodeId, gen: u32, word: u64, observed: NodeState) -> Computed {
        if word != 0 && taint_memo::memoized(word, self.runtime_id, id) {
            if is_computing(observed) {
                self.wait_on_claim(id, 1);
            }
            return self.compute_direct(id, gen);
        }
        let mut walk = Walk::new(id, observed);
        let mut root = Computed::Done;

        while let Some(Entry { id: cur, taint }) = walk.work.pop() {
            let is_root = walk.work.is_empty();
            if taint < VISIT_COMPUTING {
                self.park(ParkPoint::BeforeComputeOne(cur));
                let visit = if is_root || taint == NO_TAINT {
                    Visit::Full
                } else {
                    Visit::VerifyOnly(taint)
                };
                let computed = self.compute_one(cur, visit, is_root.then_some(gen));
                if is_root {
                    root = computed;
                } else if taint != NO_TAINT {
                    walk.taint(taint);
                }
                continue;
            }
            if walk.is_visited(cur) {
                continue;
            }

            let node = self.nodes.get(cur.0);
            let observed = if taint == VISIT {
                state::load::<C>(node.state_cell())
            } else {
                NodeState::Computing
            };
            match observed {
                NodeState::Clean => continue,
                NodeState::Computing | NodeState::ComputingDirty => {
                    if !is_root {
                        if let Some(depth) = self.own_claim_depth(cur) {
                            walk.taint(depth as u32);
                            continue;
                        }
                    }
                    // Another thread owns this subtree and is cleaning
                    // its deps itself; expanding them here would tear
                    // through a dep list mid-rewrite. Just wait for the
                    // owner, here once and then in compute_one.
                    self.wait_on_claim(cur, 1);
                    walk.visit(cur);
                }
                _ => {
                    #[cfg(test)]
                    self.expansions
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    walk.visit(cur);
                    node.for_each_dep(|dep| {
                        let dep_node = self.nodes.get(dep.0);
                        let dep_state = state::load::<C>(dep_node.state_cell());
                        if dep_state != NodeState::Clean {
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

    /// A root the taint memo holds, computed without a walk. Out of line
    /// so that the walk's own inlined `compute_one` never tests for
    /// `Visit::Direct`.
    #[cold]
    #[inline(never)]
    fn compute_direct(&self, id: NodeId, gen: u32) -> Computed {
        self.compute_one(id, Visit::Direct, Some(gen))
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
    #[inline(always)]
    fn compute_one(&self, id: NodeId, visit: Visit, gen: Option<u32>) -> Computed {
        let mut bag = Deferred::new();
        let computed = self.compute_attempts(id, visit, gen, &mut bag);
        bag.drain(self);
        computed
    }

    #[inline(always)]
    fn compute_attempts(
        &self,
        id: NodeId,
        visit: Visit,
        gen: Option<u32>,
        bag: &mut Deferred<C>,
    ) -> Computed {
        let node = self.nodes.get(id.0);
        let mut spins: u32 = 0;
        let mut g0 = gen;

        loop {
            let observed = state::load::<C>(node.state_cell());
            match observed {
                NodeState::Clean => return Computed::Done,
                NodeState::Computing | NodeState::ComputingDirty => {
                    spins += 1;
                    self.wait_on_claim(id, spins);
                    continue;
                }
                NodeState::Failed => {
                    panic!(
                        "incr-core: node {} read but its last compute panicked; it stays \
                         Failed until a dependency changes (set an upstream input to retry)",
                        id.0,
                    );
                }
                NodeState::Dead => {
                    // A bumped generation under Dead is a delete that has
                    // unhooked the node but not recycled the slot yet.
                    if g0.is_some_and(|g| g != node.generation()) {
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
                }
                NodeState::Dirty => {}
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

            self.park(ParkPoint::BeforeClaim(id));

            // Verification claims the node too, so a walk that arrives
            // meanwhile flags ComputingDirty and blocks the Clean landing.
            // The claim source tells us exactly whether this is a
            // recompute: the node may have gone Dirty since `observed`.
            let verifying = observed == NodeState::Dirty && visit != Visit::Direct;
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
                self.land(node, discard, discard);
                return Computed::Gone;
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
            let verify_rev = C::walked(&self.walk, &self.revision);

            // Red/green verification, only from Dirty: a New node has
            // nothing to verify against. It counts a dep that is not Clean
            // as changed, which catches a walk that passed this node before
            // the claim, and a dep that landed Dirty without a new stamp.
            if verifying {
                let my_verified = node.verified_at();
                let mut any_changed = false;
                node.for_each_dep(|dep| {
                    if !any_changed && !self.observed_clean(dep, my_verified) {
                        any_changed = true;
                    }
                });
                if !any_changed {
                    self.park(ParkPoint::VerifyAfterScan(id));
                    node.max_verified_at(verify_rev);
                    if self.land(node, NodeState::Clean, NodeState::Dirty) == NodeState::Clean {
                        self.record_trace(id, crate::trace::TraceAction::VerifiedClean);
                    } else {
                        self.dep_stack.poison_current_frame();
                    }
                    return Computed::Done;
                }
                if let Visit::VerifyOnly(owner) = visit {
                    // The root's own compute re-reads this node through a
                    // nested get if it still needs it, and that read
                    // poisons where needed; a poison here would cost the
                    // root a Clean landing for an edge it may have dropped.
                    self.land(node, NodeState::Dirty, NodeState::Dirty);
                    self.memoize(id, owner);
                    return Computed::Done;
                }
                // Keep the claim and recompute; the closure's nested reads
                // clean the changed deps.
            }
            let is_recompute = claimed_dirty;

            // Track deps via the strategy's dep stack. The closure runs
            // under a panic boundary: a panicking user closure must not
            // strand the node in Computing (wedging every future reader)
            // or leak its dep frame. AssertUnwindSafe is justified
            // because the only state the closure can reach is the
            // runtime's, and the Failed transition plus the frame pop
            // below restore every invariant the unwind could have
            // interrupted; no lock is held while user code runs.
            let wrote = Cell::new(0_u8);
            self.dep_stack.push_frame(id);
            // SAFETY: the closure was read under the read hold at
            // generation `g`, and the claim above was rechecked against `g`.
            // A closure is freed only by `release_claimed`, under a Dead
            // claim and after the generation bump, and no Dead claim can
            // succeed while this node is Computing. `SlotVec` never moves
            // its entries, so the `Box` still points where it was read.
            let compute = unsafe { &*compute };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compute(self, id.0, is_recompute, &wrote)
            }));
            let frame = self.dep_stack.pop_frame();
            if value_code::thread_word() & value_code::MEMO_BIT != 0 {
                taint_memo::clear_from(self.dep_stack.depth() as u32);
            }

            let value_changed = match result {
                Ok(changed) => changed,
                Err(payload) => {
                    // Publish the union of the old deps and whatever was
                    // recorded before the panic. Without the recorded
                    // subset, a first-compute panic leaves no reverse
                    // edges at all and no set() can ever flip the node
                    // out of Failed; without the old set, a recompute
                    // panic would lose recovery paths through deps the
                    // failed run had not reached yet. Extra edges only
                    // cause spurious re-verification, never staleness.
                    let landing = match self.publish_deps(id, &frame.deps, true, verify_rev, bag) {
                        // A stale dep's slot may already belong to another
                        // node, so no union is published. The old deps keep
                        // their reverse edges.
                        Publish::Stale => NodeState::New,
                        // A new dep had a change in flight when we published,
                        // and that change's walk may have read the dep's
                        // dependents before our edge existed. Nothing would
                        // ever flip us out of Failed, so land Dirty and let
                        // the next reader retry.
                        Publish::Superseded => NodeState::Dirty,
                        // verified_at was not raised, so once a dependency
                        // changes, the dirty walk flips Failed -> Dirty and
                        // red/green sees the change and recomputes for real.
                        // If a set() already flagged us mid-compute, land
                        // Dirty now so the next reader retries immediately.
                        Publish::Published => NodeState::Failed,
                    };
                    let flagged = if landing == NodeState::New {
                        NodeState::New
                    } else {
                        NodeState::Dirty
                    };
                    // The slot already holds the new value when the panic
                    // came after the write (a displaced value's Drop, or
                    // a reclaim inside its retire). Stamp it: a recompute
                    // that produces the same value cuts off, and without
                    // the stamp a dependent verified against the old
                    // value would stay Clean over the new one.
                    if wrote.get() != 0 {
                        node.max_changed_at(stamp_rev);
                    }
                    self.land(node, landing, flagged);
                    // An enclosing closure that catches the unwind has
                    // still read a dep that did not land Clean.
                    self.dep_stack.poison_current_frame();
                    bag.drain_carried(payload);
                }
            };

            let outcome = self.publish_deps(id, &frame.deps, false, verify_rev, bag);

            // The closure has already written its value into the slot, and
            // the next recompute's early cutoff compares against it, so a
            // changed value is stamped even when it is never returned.
            if value_changed || !is_recompute {
                node.max_changed_at(stamp_rev);
            }

            if outcome == Publish::Stale {
                // The value may have been read from a dep slot that was
                // released or recycled under the closure, so it is never
                // returned: land New and recompute. The retry either stops
                // reading the dep or panics on its stale handle.
                self.land(node, NodeState::New, NodeState::New);
                self.park_with_bag(ParkPoint::StaleRetry(id), bag);
                continue;
            }

            if frame.poisoned {
                // The value may be built on a result the closure
                // swallowed from a dep that did not land Clean or whose
                // cleanup failed, and that dep's stamps need not have
                // moved, so a verified_at at or above them would let the
                // next read verify this node Clean over that value.
                // Zero fails verification against every Clean dep
                // (revisions start at 1), so the next read recomputes.
                node.set_verified_at(0);
            } else {
                node.max_verified_at(verify_rev);
            }

            // Finish. Land Dirty instead of Clean if a set() flagged us
            // ComputingDirty mid-compute, if a new dep changed after we
            // read it, or if a dep we read did not itself land Clean, so
            // the next reader recomputes against the newer input.
            let target = if outcome == Publish::Superseded || frame.poisoned {
                NodeState::Dirty
            } else {
                NodeState::Clean
            };
            if self.land(node, target, NodeState::Dirty) != NodeState::Clean {
                // The enclosing compute read this node through a non-Clean
                // landing. Its own publish may take the fast path, where
                // nothing else would catch that.
                self.dep_stack.poison_current_frame();
            }
            self.record_trace(id, crate::trace::TraceAction::Recomputed { value_changed });
            return Computed::Done;
        }
    }

    /// The depth of the frame on this thread (or, under `Local`, this
    /// runtime) that holds `id`'s `Computing` claim. This is the walk's
    /// own held set; a set's claim is not on `state`, so it never shows
    /// up here.
    fn own_claim_depth(&self, id: NodeId) -> Option<usize> {
        self.dep_stack.frame_depth(id)
    }

    /// Memoize a node verify-only left non-Clean under the frame at depth
    /// `owner`, whose pop clears the entry.
    fn memoize(&self, id: NodeId, owner: u32) {
        taint_memo::memoize(self.runtime_id, id, owner);
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
    fn land(&self, node: &NodeData<C>, target: NodeState, flagged: NodeState) -> NodeState {
        if state::try_transition::<C>(node.state_cell(), NodeState::Computing, target).is_ok() {
            return target;
        }
        let owned =
            state::try_transition::<C>(node.state_cell(), NodeState::ComputingDirty, flagged);
        debug_assert!(
            owned.is_ok(),
            "only the claim's owner leaves ComputingDirty"
        );
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
        panicked: bool,
        verify_rev: u64,
        bag: &mut Deferred<C>,
    ) -> Publish {
        let node = self.nodes.get(id.0);

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
            let mut union = old_deps.clone();
            for &(d, _) in recorded {
                if !union.contains(&d) {
                    union.push(d);
                }
            }
            union
        } else {
            recorded.iter().map(|&(d, _)| d).collect()
        };

        if old_deps == new_deps {
            return Publish::Published;
        }

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
            old_deps.contains(d)
                || self.observed_clean(*d, verify_rev)
                || (panicked && self.nodes.get(d.0).state() == NodeState::Failed)
        };
        let superseded = !new_deps.iter().all(settled);

        // The displaced list is retired at the attempt's drain, after its
        // landing: a retire can reclaim pending values and run their
        // `Drop` under this hold otherwise.
        if let Some(displaced) = node.install_deps(&new_deps) {
            bag.push_dep_list(displaced);
        }

        // Linear scans for small dep sets are faster than HashSet
        // construction below ~16 items.
        for old_dep in &old_deps {
            if !new_deps.contains(old_dep) {
                inner.dependents[old_dep.0 as usize].remove(id);
            }
        }
        // A dep read twice is in `new_deps` twice, and its list already
        // ends with `id` from the first push, so each list holds `id` at
        // most once and a delete's back-to-front remove finds it.
        for new_dep in &new_deps {
            let list = &mut inner.dependents[new_dep.0 as usize];
            if !old_deps.contains(new_dep) && list.last() != Some(id) {
                list.push(id);
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
        node.state() == NodeState::Clean && node.changed_at() <= rev
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

#[cfg(test)]
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
                assert_eq!(
                    dep.state(),
                    NodeState::Clean,
                    "(I) violated: Clean node {x} has non-Clean dep {}",
                    s.0
                );
            });
        }
        for slot in 0..self.nodes.len() as usize {
            let mut ids: Vec<NodeId> = inner.dependents[slot].iter().collect();
            let n = ids.len();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), n, "dependents[{slot}] holds a node twice");
        }
        for &slot in &inner.free_nodes {
            assert!(
                inner.dependents[slot as usize].is_empty(),
                "freed slot {slot} still has reverse edges {:?}",
                inner.dependents[slot as usize]
            );
        }
    }

    pub(crate) fn has_reverse_edge<T, U>(&self, dep: Incr<T>, dependent: Incr<U>) -> bool {
        self.inner.read().dependents[dep.slot() as usize].contains(NodeId(dependent.slot()))
    }

    pub(crate) fn state_of<T>(&self, handle: Incr<T>) -> NodeState {
        self.nodes.get(handle.slot()).state()
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

    pub(crate) fn assert_walks_complete(&self) {
        assert_eq!(
            C::walked(&self.walk, &self.revision),
            self.current_revision(),
            "the walked watermark lags the revision at quiescence"
        );
    }
}

#[cfg(all(test, not(loom)))]
#[path = "runtime_race_tests.rs"]
mod race_tests;

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
        assert!(!rt.dep_stack.current_frame_active());

        rt.set(input, 5);
        assert_eq!(rt.get(parent), 12);
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
}
