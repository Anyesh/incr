//! `NodeData<C: Cells>`: the per-node read-hot struct, parameterized over
//! the [`Cells`] strategy. Production `incr-concurrent` uses a 64-byte
//! cache-line-aligned layout to keep reader traversal at one cache line
//! per node; this design carries forward unchanged into `incr-core`.
//!
//! The const-time size and alignment assertions are load-bearing under
//! both strategies. The spike validated that field-by-field the `Local`
//! cells (Cell-backed) and `Shared` cells (atomic-backed) produce the
//! same layout, so the same 64-byte total holds.
//!
//! ## Layout (both strategies)
//!
//! ```text
//! offset  size   field
//! ------  ----   -----
//!    0     8     verified_at   Cells::U64
//!    8     8     changed_at    Cells::U64
//!   16     8     overflow_deps Cells::U64   (raw pointer stored as u64 for the spike;
//!                                            full DepList-pointer machinery lands in
//!                                            the next step of the consolidation)
//!   24    28     inline_deps   [Cells::U32; 7]
//!   52     4     arena_slot    u32
//!   56     2     type_tag      u16
//!   58     1     state         Cells::State
//!   59     1     dep_count     Cells::U8    (0x80 the unrun mark, 0x40 the released mark)
//!   60     4     generation    Cells::U32
//! ```
//!
//! Total: 64 bytes, 64-byte aligned. Asserted at compile time below.
//!
//! ## What lands later
//!
//! The first incr-core slice covers the layout and the basic accessors.
//! The next slice ports the inline-7 + heap-overflow dep storage with
//! proper Drop reclamation (replacing the leaky `replace_deps_leaking_old_overflow`
//! from production with a hazard-pointer reclaimed path). The slice
//! after that lifts the segmented node store. Tracking in the
//! consolidation plan.

use crate::cells::Cells;
use crate::state::NodeState;
use haphazard::{AtomicPtr as HzAtomicPtr, Global, HazardPointer, Replaced};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const SENTINEL: NodeId = NodeId(u32::MAX);
}

/// Heap-allocated overflow dependency list. Used when a node has more
/// than seven dependencies.
///
/// Reclamation policy: when a node's dep set changes and the new list
/// requires re-allocation, the OLD overflow list is retired through
/// the `haphazard` global domain. Hazard-pointer protection in
/// [`NodeData::for_each_dep`] guarantees concurrent readers can finish
/// their traversal before the retired list is freed. Memory is
/// reclaimed during normal operation (not just at runtime drop), so
/// long-lived runtimes with churning dynamic deps no longer accumulate
/// retired lists.
pub struct DepList {
    pub(crate) deps: Box<[NodeId]>,
}

/// Holder for a node's overflow [`DepList`], selected per strategy
/// through `Cells::DepListCell` so the loom strategy can swap out the
/// hazard-pointer cell that `Local` and `Shared` share.
pub trait DepListCell: 'static {
    /// What [`Self::replace`] hands back for the displaced list. Dropping
    /// it frees the list once no reader can still be iterating it.
    type Retired;
    fn new_null() -> Self;
    /// Install `list` (or null) and hand back the displaced list's retire
    /// token. The retire itself may run user code (haphazard reclaims
    /// pending values inside any retire), so the caller drops the token
    /// at a safe point, never under an incr lock or before its landing.
    /// The caller must be the node's single writer, which the state
    /// machine grants.
    fn replace(&self, list: Option<Box<DepList>>) -> Option<Self::Retired>;
    /// Run `f` on the installed list and return true, or return false
    /// without calling `f` when none is installed. The caller must have
    /// observed an overflow `dep_count` with Acquire; a null list under
    /// that count is an inline install in progress.
    fn with_list(&self, f: &mut dyn FnMut(&DepList)) -> bool;
}

/// The retire token of a dep list displaced from an [`HzDepListCell`].
/// Dropping it retires the list through the global haphazard domain,
/// which frees it once no reader in `with_list` still references it.
pub struct RetiredDepList(Replaced<DepList, Global, Box<DepList>>);

impl Drop for RetiredDepList {
    fn drop(&mut self) {
        // SAFETY: the pointer came from a Box installed by a previous
        // replace, was displaced exactly once by the swap that made this
        // token, and is never loaded from the cell again.
        unsafe { self.0.retire() };
    }
}

/// The retire token type of `C`'s dep-list cell.
pub(crate) type RetiredDeps<C> = <<C as Cells>::DepListCell as DepListCell>::Retired;

/// Hazard-pointer overflow cell used by `Local` and `Shared`.
pub struct HzDepListCell(HzAtomicPtr<DepList>);

impl DepListCell for HzDepListCell {
    type Retired = RetiredDepList;

    fn new_null() -> Self {
        // SAFETY: a null pointer is always a valid initial value.
        Self(unsafe { HzAtomicPtr::new(std::ptr::null_mut()) })
    }

    fn replace(&self, list: Option<Box<DepList>>) -> Option<RetiredDepList> {
        let displaced = match list {
            Some(list) => self.0.swap(list),
            // SAFETY: swap_ptr with a null target is safe.
            None => unsafe { self.0.swap_ptr(std::ptr::null_mut()) },
        };
        displaced.map(RetiredDepList)
    }

    fn with_list(&self, f: &mut dyn FnMut(&DepList)) -> bool {
        let mut hazard = HazardPointer::new();
        // SAFETY: the pointer is populated by replace with Box-allocated
        // DepLists; retirements go through the global haphazard domain,
        // so the reference stays valid for the lifetime of `hazard`.
        let list: Option<&DepList> = unsafe { self.0.load(&mut hazard) };
        match list {
            Some(list) => {
                f(list);
                true
            }
            None => false,
        }
    }
}

impl Drop for HzDepListCell {
    fn drop(&mut self) {
        // For the runtime-drop case all hazards are already gone, so the
        // retire reclaims immediately.
        drop(self.replace(None));
    }
}

#[repr(C, align(64))]
pub struct NodeData<C: Cells> {
    pub(crate) verified_at: C::U64,
    pub(crate) changed_at: C::U64,
    pub(crate) overflow_deps: C::DepListCell,
    pub(crate) inline_deps: [C::U32; 7],
    // Cells (not plain fields) so delete_node can recycle a slot in
    // place through &self: a recycled slot may host a different value
    // type and points at a different arena slot.
    pub(crate) arena_slot: C::U32,
    pub(crate) type_tag: C::U16,
    pub(crate) state: C::State,
    dep_count: C::U8,
    pub(crate) generation: C::U32,
}

const _: () = assert!(crate::bits::disjoint(&[
    NodeData::<crate::cells::Local>::MEMO,
    NodeData::<crate::cells::Local>::SET_CLAIM,
    NodeData::<crate::cells::Local>::DELETED,
]));
const _: () = assert!(crate::bits::disjoint(&[
    NodeData::<crate::cells::Local>::COUNT_MASK as u64,
    NodeData::<crate::cells::Local>::RELEASED as u64,
    NodeData::<crate::cells::Local>::UNRUN as u64,
]));

impl<C: Cells> NodeData<C> {
    /// The `changed_at` of an input created, recycled or set at revision
    /// `revision`: revisions are even, inputs carry the odd stamp below
    /// and computes the even revision they ran at, so a stamp just below
    /// every later claim's revision is free for a seed.
    #[inline(always)]
    pub(crate) fn input_stamp(revision: u64) -> u64 {
        revision - 1
    }

    /// Construct a new input node. Input nodes start `Clean` because their
    /// value is provided at creation. `revision` seeds `verified_at`, and
    /// `changed_at` gets `revision - 1`.
    pub fn new_input(type_tag: u16, arena_slot: u32, revision: u64) -> Self {
        Self {
            verified_at: C::new_u64(revision),
            changed_at: C::new_u64(Self::input_stamp(revision)),
            overflow_deps: C::DepListCell::new_null(),
            inline_deps: Self::empty_inline_deps(),
            arena_slot: C::new_u32(arena_slot),
            type_tag: C::new_u16(type_tag),
            state: C::new_state(NodeState::Clean.as_u8()),
            dep_count: C::new_u8(0),
            generation: C::new_u32(0),
        }
    }

    /// Construct a new query node. Query nodes start `New` because their
    /// value has not been computed; the first reader CASes to `Computing`
    /// and runs the compute closure.
    pub fn new_query(type_tag: u16, arena_slot: u32) -> Self {
        Self {
            verified_at: C::new_u64(0),
            changed_at: C::new_u64(0),
            overflow_deps: C::DepListCell::new_null(),
            inline_deps: Self::empty_inline_deps(),
            arena_slot: C::new_u32(arena_slot),
            type_tag: C::new_u16(type_tag),
            state: C::new_state(NodeState::New.as_u8()),
            dep_count: C::new_u8(0),
            generation: C::new_u32(0),
        }
    }

    /// Tear down a node for deletion: clear deps and Release-store Dead.
    /// The caller (`release_claimed`) owns the transition; the slot stays
    /// allocated and is reinitialized by [`Self::reinit_recycled`]. The
    /// displaced dep list's retire token is handed back, to be dropped
    /// after the delete has released `inner`.
    #[must_use]
    pub(crate) fn retire_for_delete(&self) -> Option<RetiredDeps<C>> {
        let displaced = self.install_deps(&[]);
        C::state_store_release(&self.state, NodeState::Dead.as_u8());
        displaced
    }

    /// Move the slot to a new generation so handles to the deleted node
    /// are rejected. Must run under `inner.write()`, after the reverse
    /// edges naming this slot are unhooked, so no `dependents` list ever
    /// pairs this slot with a generation it does not carry.
    pub(crate) fn bump_generation(&self) {
        C::u32_store_relaxed(&self.generation, self.generation().wrapping_add(1));
    }

    /// Reinitialize a recycled (Dead) slot in place for a new node.
    /// Field stores are relaxed; the final Release store on `state` is
    /// the publication point, and pairs with readers' Acquire state
    /// loads. The generation was already bumped by the delete, so
    /// handles minted now carry the new generation and stale ones fail
    /// the check. The Release fence before the first field store pairs
    /// with the Acquire fence a lock-free reader issues between its
    /// field loads and its generation compare, so a reader that saw any
    /// of the new fields also sees the bump.
    pub(crate) fn reinit_recycled(
        &self,
        type_tag: u16,
        arena_slot: u32,
        revision: u64,
        is_input: bool,
    ) {
        C::fence_release();
        C::u64_store_relaxed(&self.verified_at, if is_input { revision } else { 0 });
        C::u64_store_relaxed(
            &self.changed_at,
            if is_input {
                Self::input_stamp(revision)
            } else {
                0
            },
        );
        C::u32_store_relaxed(&self.arena_slot, arena_slot);
        C::u16_store_relaxed(&self.type_tag, type_tag);
        C::u8_store_relaxed(&self.dep_count, 0);
        let initial = if is_input {
            NodeState::Clean
        } else {
            NodeState::New
        };
        C::state_store_release(&self.state, initial.as_u8());
    }

    #[inline(always)]
    pub fn arena_slot(&self) -> u32 {
        C::u32_load_relaxed(&self.arena_slot)
    }

    #[inline(always)]
    pub fn type_tag(&self) -> u16 {
        C::u16_load_relaxed(&self.type_tag)
    }

    #[inline(always)]
    pub fn state(&self) -> NodeState {
        NodeState::from_u8(C::state_load_acquire(&self.state))
    }

    #[inline(always)]
    pub fn state_cell(&self) -> &C::State {
        &self.state
    }

    /// The revision stamp, without an input's [`Self::SET_CLAIM`] and
    /// [`Self::DELETED`] bits.
    #[inline(always)]
    pub fn verified_at(&self) -> u64 {
        C::u64_load_acquire(&self.verified_at) & Self::REVISION_BITS
    }

    #[inline(always)]
    pub fn changed_at(&self) -> u64 {
        C::u64_load_acquire(&self.changed_at)
    }

    #[inline(always)]
    pub fn set_verified_at(&self, v: u64) {
        C::u64_store_release(&self.verified_at, v);
    }

    #[inline(always)]
    pub fn set_changed_at(&self, v: u64) {
        C::u64_store_release(&self.changed_at, v);
    }

    /// Raise `verified_at` to `v` if `v` is larger. Protocol stamps must
    /// use max, not store: two threads can race to stamp (a verifier and
    /// a computer) and a plain store could regress the timestamp, masking
    /// a change from a later verifier.
    #[inline(always)]
    pub fn max_verified_at(&self, v: u64) {
        let cur = C::u64_load_relaxed(&self.verified_at);
        if C::SINGLE_THREADED {
            C::u64_store_relaxed(&self.verified_at, (cur & !Self::MEMO).max(v));
        } else if cur & Self::MEMO != 0 {
            self.stamp_over_memo(cur, v);
        } else {
            C::u64_fetch_max(&self.verified_at, v);
        }
    }

    /// [`Self::max_verified_at`] over a memo, which the stamp ends. Only the
    /// claim's holder sets a memo, so none appears between the load and
    /// the max.
    #[cold]
    #[inline(never)]
    fn stamp_over_memo(&self, mut cur: u64, v: u64) {
        loop {
            let new = (cur & !Self::MEMO).max(v);
            match C::u64_compare_exchange(&self.verified_at, cur, new) {
                Ok(_) => return,
                Err(seen) => cur = seen,
            }
        }
    }

    /// Set on an input's `verified_at` while a `set` holds the input. The
    /// claim lives here, not on `state`, because no reader of an input
    /// loads its `verified_at` (only a query verifying itself reads its
    /// own), so a set's claim and release are independent of every read.
    pub const SET_CLAIM: u64 = 1 << 62;

    /// Set on `verified_at` by the delete that claimed the node `Dead`,
    /// so a later set claim fails. Cleared only by the recycle's store.
    pub const DELETED: u64 = 1 << 63;

    /// Set on a Dirty query's `verified_at` while it holds a memo for the
    /// revision in its `changed_at` (`set_memo`). Masked out of the stamp,
    /// so verification reads 0, never verified; any later stamp ends it.
    pub const MEMO: u64 = 1 << 61;

    const REVISION_BITS: u64 = !(Self::SET_CLAIM | Self::DELETED | Self::MEMO);

    /// Take a `set`'s claim on this input. Returns the stamp it claimed
    /// over, for [`Self::release_set_claim`], or the word holding
    /// [`Self::SET_CLAIM`] or [`Self::DELETED`] that refused it.
    #[inline(always)]
    pub(crate) fn try_claim_set(&self) -> Result<u64, u64> {
        let mut cur = C::u64_load_acquire(&self.verified_at);
        loop {
            if cur & !Self::REVISION_BITS != 0 {
                return Err(cur);
            }
            match C::u64_compare_exchange(&self.verified_at, cur, cur | Self::SET_CLAIM) {
                Ok(_) => return Ok(cur),
                Err(seen) => cur = seen,
            }
        }
    }

    /// Release a claim taken over `base`, raising the stamp to `rev` in
    /// the same RMW. An add rather than a store, so a concurrent delete's
    /// [`Self::DELETED`] survives: under the claim no other write moves
    /// the stamp bits (a max loses to the claim bit), so they still read
    /// `base`, and `rev >= base` keeps the add from carrying into the
    /// flag bits.
    #[inline(always)]
    pub(crate) fn release_set_claim(&self, base: u64, rev: u64) {
        debug_assert!(base <= rev && rev & !Self::REVISION_BITS == 0);
        C::u64_fetch_add(
            &self.verified_at,
            (rev - base).wrapping_sub(Self::SET_CLAIM),
        );
    }

    /// Mark the node deleted, so no `set` claims it from here on. Returns
    /// true if a set holds it now; the delete must wait that set out
    /// with [`Self::set_claimed`] before it retires the node.
    #[inline(always)]
    pub(crate) fn mark_deleted(&self) -> bool {
        C::u64_fetch_or(&self.verified_at, Self::DELETED) & Self::SET_CLAIM != 0
    }

    #[inline(always)]
    pub(crate) fn set_claimed(&self) -> bool {
        C::u64_load_acquire(&self.verified_at) & Self::SET_CLAIM != 0
    }

    /// Raise `changed_at` to `v` if `v` is larger. See [`Self::max_verified_at`].
    /// Clears a memo first: a landing raises `changed_at` before it
    /// rewrites `verified_at`, and the old memo must not read as one for
    /// the new `changed_at`.
    #[inline(always)]
    pub fn max_changed_at(&self, v: u64) {
        let cur = C::u64_load_relaxed(&self.verified_at);
        if C::SINGLE_THREADED {
            C::u64_store_relaxed(&self.verified_at, cur & !Self::MEMO);
        } else if cur & Self::MEMO != 0 {
            self.stamp_over_memo(cur, 0);
        }
        C::u64_fetch_max(&self.changed_at, v);
    }

    /// Keep a Dirty query's value for the rest of revision `rev`: reads in
    /// `rev` take it without a run, and the first read after a set runs
    /// the closure. Raising `changed_at` to `rev` only makes the node
    /// newer to its readers' verification. Before the landing's store, so
    /// a reader that loads the Dirty state sees the memo.
    #[inline(always)]
    pub(crate) fn set_memo(&self, rev: u64) {
        self.set_verified_at(Self::MEMO);
        C::u64_fetch_max(&self.changed_at, rev);
    }

    /// Whether the query, loaded Dirty, holds a memo for the revision
    /// `revision` loads, which only a memo's node loads.
    #[inline(always)]
    pub(crate) fn memo_at(&self, revision: impl FnOnce() -> u64) -> bool {
        // The bit is loaded again after `changed_at`: a `changed_at` from a
        // later landing orders the clear before it.
        C::u64_load_acquire(&self.verified_at) & Self::MEMO != 0 && {
            let changed = self.changed_at();
            C::u64_load_acquire(&self.verified_at) & Self::MEMO != 0 && changed == revision()
        }
    }

    /// Sentinel meaning "a set() is in flight on this input". Any
    /// verifier comparing `changed_at > my_verified` sees the marker as
    /// infinitely new and recomputes instead of verifying clean. Without
    /// it there is a mask window: a setter that has bumped the global
    /// revision but not yet stamped `changed_at` lets a concurrent
    /// verifier stamp `verified_at` at or above the setter's revision
    /// while having read the old stamp, after which the change is
    /// invisible forever.
    pub const CHANGED_PENDING: u64 = u64::MAX;

    /// Enter the pending-marker state. Must happen BEFORE the value swap
    /// and revision bump in `set()`; see [`Self::CHANGED_PENDING`].
    #[inline(always)]
    pub fn mark_changed_pending(&self) {
        C::u64_fetch_max(&self.changed_at, Self::CHANGED_PENDING);
    }

    /// Replace the pending marker with the setter's concrete revision,
    /// without regressing a concurrent setter's higher revision and
    /// without erasing a concurrent setter's fresh marker more than
    /// necessary (max semantics among concrete revisions).
    pub fn settle_changed_at(&self, r: u64) {
        let mut cur = C::u64_load_acquire(&self.changed_at);
        loop {
            let target = if cur == Self::CHANGED_PENDING || r > cur {
                r
            } else {
                return;
            };
            match C::u64_compare_exchange(&self.changed_at, cur, target) {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Dep-count cell value: exact count for 0..=7 (inline storage),
    /// [`Self::OVERFLOW_MARKER`] when deps live in the overflow list
    /// (whose length is authoritative).
    #[inline(always)]
    pub fn dep_count(&self) -> u8 {
        Self::count_of(C::u8_load_relaxed(&self.dep_count))
    }

    /// The count in a `dep_count` cell value, without its marks.
    #[inline(always)]
    fn count_of(raw: u8) -> u8 {
        raw & Self::COUNT_MASK
    }

    #[inline(always)]
    fn unrun_bit(&self) -> u8 {
        if C::SINGLE_THREADED {
            0
        } else {
            C::u8_load_relaxed(&self.dep_count) & Self::UNRUN
        }
    }

    /// Whether the node's claim was last released unrun. Always false
    /// under a single-threaded strategy, where no other thread waits.
    #[inline(always)]
    pub(crate) fn unrun(&self) -> bool {
        self.unrun_bit() != 0
    }

    /// Records whether the node's claim is being released unrun. Only the
    /// claim's owner calls it, before the release that publishes it.
    #[inline(always)]
    pub(crate) fn set_unrun(&self, unrun: bool) {
        if !C::SINGLE_THREADED {
            self.set_mark(Self::UNRUN, unrun);
        }
    }

    /// Whether the runtime holds a released entry for the node
    /// (`Inner::released_deps`). An install keeps it.
    #[inline(always)]
    pub(crate) fn released_mark(&self) -> bool {
        C::u8_load_relaxed(&self.dep_count) & Self::RELEASED != 0
    }

    /// Only the claim's owner calls it, under the hold that sets or drops
    /// the entry.
    #[inline(always)]
    pub(crate) fn set_released_mark(&self, released: bool) {
        self.set_mark(Self::RELEASED, released);
    }

    /// The unrun and released marks, which an install keeps.
    #[inline(always)]
    fn marks(&self) -> u8 {
        C::u8_load_relaxed(&self.dep_count) & !Self::COUNT_MASK
    }

    #[inline(always)]
    fn set_mark(&self, bit: u8, on: bool) {
        let raw = C::u8_load_relaxed(&self.dep_count);
        let marked = if on { raw | bit } else { raw & !bit };
        if marked != raw {
            // Release like install_deps's, so a scan that reads this
            // value still sees the deps installed before it.
            C::u8_store_release(&self.dep_count, marked);
        }
    }

    #[inline(always)]
    pub fn generation(&self) -> u32 {
        C::u32_load_relaxed(&self.generation)
    }

    #[inline(always)]
    pub fn set_state(&self, v: u8) {
        C::state_store_release(&self.state, v);
    }

    /// Install a new dep list. Inline-7 path stores into the inline
    /// array; overflow path heap-allocates a `DepList` and Release-stores
    /// the pointer.
    ///
    /// Reclamation: any displaced overflow list's retire token is handed
    /// back, and dropping it retires the list through `C::DepListCell`,
    /// which under `Local` and `Shared` is the `haphazard` global domain.
    /// Concurrent readers in [`Self::for_each_dep`] hold a
    /// `HazardPointer` while dereferencing the slot, so the retired list
    /// is not freed until every protecting reader has finished. Nothing
    /// in here retires, so nothing in here can panic between the swap
    /// and the count store.
    #[must_use]
    pub(crate) fn install_deps(&self, new_deps: &[NodeId]) -> Option<RetiredDeps<C>> {
        let count = new_deps.len();
        if count <= 7 {
            // Each id is Release-stored on its own, not just covered by
            // the count's Release below: a scan can read a previous
            // install's equal count and still read this id, and it must
            // then see everything this writer knew, the id's segment
            // allocation included.
            for (i, dep) in new_deps.iter().enumerate() {
                C::u32_store_release(&self.inline_deps[i], dep.0);
            }
            // Going inline: displace any stale overflow list.
            // for_each_dep's count<=7 path never reads overflow_deps, but
            // keeping the stale list would hold its memory until the next
            // overflow install or the node's drop. A scan that read the
            // overflow marker before this store finds the null and
            // retries on the count stored next.
            let displaced = self.overflow_deps.replace(None);
            C::u8_store_release(&self.dep_count, count as u8 | self.marks());
            displaced
        } else {
            let list = Box::new(DepList {
                deps: new_deps.to_vec().into_boxed_slice(),
            });
            let displaced = self.overflow_deps.replace(Some(list));
            // The cell is one byte, so counts above 7 are encoded as a
            // marker and the DepList length is authoritative. This is
            // what lifts the old 255-dep ceiling: a query may read any
            // number of inputs.
            C::u8_store_release(&self.dep_count, Self::OVERFLOW_MARKER | self.marks());
            displaced
        }
    }

    /// Sentinel stored in `dep_count` when the dep list lives in
    /// `overflow_deps`. Any value above 7 would do; 8 is the smallest.
    pub const OVERFLOW_MARKER: u8 = 8;

    /// `dep_count` bit for [`Self::unrun`], kept there because the
    /// claim's owner writes both and its release publishes both.
    const UNRUN: u8 = 0x80;

    /// `dep_count` bit for [`Self::released_mark`]: `NodeData` fills its
    /// line, and a publish already reads this cell.
    const RELEASED: u8 = 0x40;

    const COUNT_MASK: u8 = 0x0f;

    /// Iterate over the node's recorded dependencies. A scan may race an
    /// `install_deps` on another thread (the dirty walk scans without a
    /// claim); it then sees some install's count and, per id, some
    /// install's id, and every id it sees names a node whose creation
    /// happens-before the scan.
    ///
    /// Up to 7 deps live inline; beyond that, `overflow_deps` points at
    /// a heap-allocated `DepList` whose load is protected (by a
    /// `HazardPointer` under `Local` and `Shared`). A concurrent retire by an `install_deps`
    /// writer will defer the actual free until this reader's hazard
    /// is released.
    ///
    /// The dispatch is intentionally split: the inline fast path is
    /// `#[inline]`-friendly and stays small enough to be inlined into
    /// the caller. The cold overflow path is `#[inline(never)]` so
    /// `HazardPointer::new` (thread_local lookup + potential allocation)
    /// is not duplicated into every call site.
    #[inline]
    pub fn for_each_dep(&self, mut f: impl FnMut(NodeId)) {
        // Acquire pairs with install_deps's Release store on dep_count:
        // a reader that observes the new count must also observe the
        // inline entries (or the overflow pointer) written before it.
        // Relaxed here could yield a count ahead of the entries and index
        // sentinel garbage.
        let count = Self::count_of(C::u8_load_acquire(&self.dep_count));
        if count <= 7 {
            self.for_each_inline_dep(count, &mut f);
        } else {
            self.for_each_overflow_dep(&mut f);
        }
    }

    /// The deps as [`Self::for_each_dep`] scans them when they fit inline,
    /// or `None` when they overflow. A caller's loop body over these stays
    /// inline where a closure handed to `for_each_dep` may not, as its
    /// overflow arm needs an out-of-line copy.
    #[inline(always)]
    pub fn inline_deps(&self) -> Option<impl Iterator<Item = NodeId> + '_> {
        let count = Self::count_of(C::u8_load_acquire(&self.dep_count));
        (count <= 7).then(move || {
            (0..count as usize).map(move |i| NodeId(C::u32_load_acquire(&self.inline_deps[i])))
        })
    }

    #[inline(always)]
    fn for_each_inline_dep<F: FnMut(NodeId) + ?Sized>(&self, count: u8, f: &mut F) {
        for i in 0..(count as usize) {
            // Acquire pairs with install_deps's Release store of the id,
            // so the scan has happens-before with the id's creation even
            // when the count it read predates the install.
            let raw = C::u32_load_acquire(&self.inline_deps[i]);
            f(NodeId(raw));
        }
    }

    /// A null list under the overflow marker is an inline install that
    /// has swapped the list out and not yet stored its count: a scan
    /// without a claim can see that window. The count is reloaded and
    /// the scan retried until it lands on either arm; the installer
    /// stores the inline count right after the swap, so the retry ends.
    #[inline(never)]
    fn for_each_overflow_dep(&self, f: &mut dyn FnMut(NodeId)) {
        let mut spins = 0;
        loop {
            let scanned = self.overflow_deps.with_list(&mut |list| {
                for &id in list.deps.iter() {
                    f(id);
                }
            });
            if scanned {
                return;
            }
            C::backoff(spins);
            spins += 1;
            let count = Self::count_of(C::u8_load_acquire(&self.dep_count));
            if count <= 7 {
                self.for_each_inline_dep(count, f);
                return;
            }
        }
    }

    fn empty_inline_deps() -> [C::U32; 7] {
        [
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
            C::new_u32(NodeId::SENTINEL.0),
        ]
    }
}

const _: () = {
    type N = NodeData<crate::cells::Local>;
    assert!(
        N::OVERFLOW_MARKER & !N::COUNT_MASK == 0,
        "dep_count's overflow marker lies outside its count bits"
    );
};
const _: () = assert!(
    std::mem::size_of::<NodeData<crate::cells::Local>>() == 64,
    "NodeData<Local> must be exactly one 64-byte cache line"
);
const _: () = assert!(
    std::mem::align_of::<NodeData<crate::cells::Local>>() == 64,
    "NodeData<Local> must be 64-byte aligned"
);
const _: () = assert!(
    std::mem::size_of::<NodeData<crate::cells::Shared>>() == 64,
    "NodeData<Shared> must be exactly one 64-byte cache line"
);
const _: () = assert!(
    std::mem::align_of::<NodeData<crate::cells::Shared>>() == 64,
    "NodeData<Shared> must be 64-byte aligned"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    #[test]
    fn local_node_is_64_bytes() {
        assert_eq!(std::mem::size_of::<NodeData<Local>>(), 64);
        assert_eq!(std::mem::align_of::<NodeData<Local>>(), 64);
    }

    #[test]
    fn shared_node_is_64_bytes() {
        assert_eq!(std::mem::size_of::<NodeData<Shared>>(), 64);
        assert_eq!(std::mem::align_of::<NodeData<Shared>>(), 64);
    }

    #[test]
    fn local_input_roundtrip() {
        let n: NodeData<Local> = NodeData::new_input(0, 42, 8);
        assert_eq!(n.arena_slot(), 42);
        assert_eq!(n.verified_at(), 8);
        assert_eq!(n.changed_at(), 7);
        assert_eq!(n.state(), NodeState::Clean);
        n.set_verified_at(11);
        assert_eq!(n.verified_at(), 11);
    }

    #[test]
    fn shared_input_roundtrip() {
        let n: NodeData<Shared> = NodeData::new_input(0, 42, 8);
        assert_eq!(n.arena_slot(), 42);
        assert_eq!(n.verified_at(), 8);
        assert_eq!(n.changed_at(), 7);
        assert_eq!(n.state(), NodeState::Clean);
        n.set_verified_at(11);
        assert_eq!(n.verified_at(), 11);
    }

    #[test]
    fn local_query_starts_new() {
        let n: NodeData<Local> = NodeData::new_query(0, 99);
        assert_eq!(n.state(), NodeState::New);
        assert_eq!(n.dep_count(), 0);
    }

    #[test]
    fn shared_query_starts_new() {
        let n: NodeData<Shared> = NodeData::new_query(0, 99);
        assert_eq!(n.state(), NodeState::New);
        assert_eq!(n.dep_count(), 0);
    }

    #[test]
    fn max_stamps_are_monotonic() {
        let n: NodeData<Shared> = NodeData::new_input(0, 0, 10);
        n.max_verified_at(7);
        assert_eq!(n.verified_at(), 10);
        n.max_verified_at(12);
        assert_eq!(n.verified_at(), 12);
        n.max_changed_at(11);
        assert_eq!(n.changed_at(), 11);
        n.max_changed_at(5);
        assert_eq!(n.changed_at(), 11);
    }

    fn memo_ends_at_a_changed_at_raise<C: Cells>() {
        let n: NodeData<C> = NodeData::new_query(0, 99);
        n.set_memo(5);
        assert!(n.memo_at(|| 5));
        assert_eq!(n.verified_at(), 0);
        n.max_changed_at(6);
        assert!(!n.memo_at(|| 6), "a landing's changed_at kept the old memo");
        n.set_memo(6);
        assert!(n.memo_at(|| 6));
        n.max_verified_at(6);
        assert!(!n.memo_at(|| 6));
        assert_eq!(n.verified_at(), 6);
    }

    #[test]
    fn memo_ends_at_a_changed_at_raise_local() {
        memo_ends_at_a_changed_at_raise::<Local>();
    }

    #[test]
    fn memo_ends_at_a_changed_at_raise_shared() {
        memo_ends_at_a_changed_at_raise::<Shared>();
    }

    fn set_claim_and_delete_mark<C: Cells>() {
        let n: NodeData<C> = NodeData::new_input(0, 0, 10);
        let base = n.try_claim_set().unwrap();
        assert_eq!(base, 10);
        assert!(n.set_claimed());
        assert_eq!(n.verified_at(), 10, "the claim bit leaks into the stamp");
        n.max_verified_at(12);
        assert!(n.try_claim_set().is_err(), "a second claim went through");
        n.release_set_claim(base, 15);
        assert!(!n.set_claimed());
        assert_eq!(n.verified_at(), 15);

        let base = n.try_claim_set().unwrap();
        assert!(n.mark_deleted(), "the delete missed the held claim");
        n.release_set_claim(base, base);
        assert!(!n.set_claimed());
        assert_eq!(n.verified_at(), 15);
        let refused = n.try_claim_set().unwrap_err();
        assert_ne!(refused & NodeData::<C>::DELETED, 0);

        n.reinit_recycled(0, 0, 20, true);
        assert_eq!(n.try_claim_set(), Ok(20), "the recycle kept a flag bit");
        n.release_set_claim(20, 20);
        assert!(!n.mark_deleted());
    }

    #[test]
    fn local_set_claim_and_delete_mark() {
        set_claim_and_delete_mark::<Local>();
    }

    #[test]
    fn shared_set_claim_and_delete_mark() {
        set_claim_and_delete_mark::<Shared>();
    }

    #[test]
    fn deps_beyond_255_roundtrip() {
        for_both_strategies_deps(300);
        for_both_strategies_deps(8);
        for_both_strategies_deps(7);
    }

    fn for_both_strategies_deps(n: u32) {
        let local: NodeData<Local> = NodeData::new_query(0, 0);
        let shared: NodeData<Shared> = NodeData::new_query(0, 0);
        let deps: Vec<NodeId> = (0..n).map(NodeId).collect();
        drop(local.install_deps(&deps));
        drop(shared.install_deps(&deps));
        for node_deps in [collect_deps(&local) as Vec<NodeId>, collect_deps(&shared)] {
            assert_eq!(node_deps, deps, "roundtrip failed for n={}", n);
        }
        if n > 7 {
            assert_eq!(local.dep_count(), NodeData::<Local>::OVERFLOW_MARKER);
        }
    }

    fn collect_deps<C: crate::cells::Cells>(node: &NodeData<C>) -> Vec<NodeId> {
        let mut out = Vec::new();
        node.for_each_dep(|d| out.push(d));
        out
    }

    #[test]
    fn shrink_from_overflow_back_to_inline() {
        let n: NodeData<Local> = NodeData::new_query(0, 0);
        let big: Vec<NodeId> = (0..20).map(NodeId).collect();
        drop(n.install_deps(&big));
        assert_eq!(collect_deps(&n), big);
        let small: Vec<NodeId> = (0..3).map(NodeId).collect();
        drop(n.install_deps(&small));
        assert_eq!(collect_deps(&n), small);
        assert_eq!(n.dep_count(), 3);
    }

    #[test]
    fn the_unrun_mark_leaves_the_deps_alone() {
        for len in [3, 20] {
            let n: NodeData<Shared> = NodeData::new_query(0, 0);
            let deps: Vec<NodeId> = (0..len).map(NodeId).collect();
            drop(n.install_deps(&deps));
            let count = n.dep_count();
            n.set_unrun(true);
            assert!(n.unrun());
            assert_eq!(n.dep_count(), count);
            assert_eq!(collect_deps(&n), deps);
            let fewer: Vec<NodeId> = (0..2).map(NodeId).collect();
            drop(n.install_deps(&fewer));
            assert!(n.unrun(), "an install keeps the mark");
            assert_eq!(collect_deps(&n), fewer);
            n.set_unrun(false);
            assert!(!n.unrun());
            assert_eq!(n.dep_count(), 2);
        }
        let n: NodeData<Local> = NodeData::new_query(0, 0);
        drop(n.install_deps(&[NodeId(1), NodeId(2)]));
        n.set_unrun(true);
        assert!(!n.unrun(), "a single-threaded node keeps no mark");
        assert_eq!(n.dep_count(), 2);
    }

    #[test]
    fn the_released_mark_leaves_the_deps_alone() {
        fn check<C: crate::cells::Cells>() {
            for len in [3, 20] {
                let n: NodeData<C> = NodeData::new_query(0, 0);
                let deps: Vec<NodeId> = (0..len).map(NodeId).collect();
                drop(n.install_deps(&deps));
                let count = n.dep_count();
                n.set_released_mark(true);
                n.set_unrun(true);
                assert!(n.released_mark());
                assert_eq!(n.dep_count(), count);
                assert_eq!(collect_deps(&n), deps);
                n.set_unrun(false);
                assert!(n.released_mark(), "the unrun mark leaves it alone");
                let fewer: Vec<NodeId> = (0..2).map(NodeId).collect();
                drop(n.install_deps(&fewer));
                assert!(n.released_mark(), "an install keeps it");
                assert_eq!(collect_deps(&n), fewer);
                drop(n.install_deps(&deps));
                n.set_released_mark(false);
                assert!(!n.released_mark());
                assert_eq!(n.dep_count(), count);
            }
        }
        check::<Local>();
        check::<Shared>();
    }
}
