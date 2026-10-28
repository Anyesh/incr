//! Per-thread cycle state shared by every runtime on the thread: the
//! claims this thread holds outside its frame stack (the held map), the
//! engine entry depth, the kind of an engine unwind in flight, the last
//! cycle an uncaught cycle panic reported, and the roots whose episodes
//! dissolved during the current outermost entry.

use std::borrow::{Borrow, BorrowMut};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::ops::{Deref, DerefMut};

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use crate::cycle::{Cycle, Erased, ErasedStrategy};
use crate::deferred::Unbuilt;
use crate::fx::FxBuildHasher;
use crate::handle::RuntimeId;
use crate::node::NodeId;
use crate::replay;
use crate::runtime::GroupInfo;
use crate::value_code;

/// The frame whose unwind releases a claim: the frame it stands for, or
/// the frame below that its deferral passed to. A deferral to a root below
/// that frame leaves the claim anchored lower, so the anchor alone would
/// let an unwind between them leak it.
#[derive(Clone, Default)]
pub(crate) struct Owner {
    /// The frame's position.
    pos: u32,
    /// The frame's node, `u32::MAX` for any: another frame at that
    /// position later, a sibling's, owns none of it.
    node: u32,
    /// The owner is a virtual frame, the run of `node` that a doom
    /// discarded, with that frame's path. Another run of the node at the
    /// position, as its reader's retry makes when it reads the node more
    /// than once, has another path and must not take or release what the
    /// doomed run deferred.
    virtual_frame: Option<u64>,
    /// For a virtual owner, the position and path of each virtual frame
    /// below it that has popped since: a run at such a position along
    /// another path is not the owner's ancestor.
    ancestry: Vec<(u32, u64)>,
}

impl Owner {
    fn new(pos: u32, node: u32, virtual_frame: Option<u64>) -> Self {
        Self {
            pos,
            node,
            virtual_frame,
            ancestry: Vec::new(),
        }
    }

    pub(crate) fn node(&self) -> u32 {
        self.node
    }

    pub(crate) fn virtual_frame(&self) -> Option<u64> {
        self.virtual_frame
    }

    /// The run at `cidx`, of `node` (`u32::MAX` for any), at `path` (the
    /// path when a virtual frame at or above `cidx` owns claims), owns the
    /// claims of this owner: those owned at its position by it, or owned
    /// above it in its subtree.
    pub(crate) fn owned_by(&self, cidx: u32, node: u32, path: Option<u64>) -> bool {
        let at = |p: u64| Some(p) == path;
        let of_node = node == u32::MAX || self.node == node || self.node == u32::MAX;
        match self.virtual_frame {
            _ if self.pos < cidx => false,
            None => self.pos > cidx || of_node,
            Some(p) if self.pos == cidx => at(p) && of_node,
            Some(_) => self
                .ancestry
                .iter()
                .find(|a| a.0 == cidx)
                .is_none_or(|a| at(a.1)),
        }
    }
}

/// A claim this thread holds outside its frame stack: a node deferred
/// during discovery, or a member of a running episode.
pub(crate) struct Held {
    pub(crate) generation: u32,
    /// The conceptual-stack position of the root that resolves it.
    anchor: u32,
    /// The claim's owner while it is out of the held map; inside it, the
    /// claim's owner group holds it.
    owner: Owner,
    /// The owner group inside the held map, 0 outside it.
    group: u64,
    /// The claim's index in its group's members.
    group_ix: usize,
    /// The node has a Fallback or Fixpoint strategy.
    pub(crate) seeded: bool,
    /// The claim was taken from New.
    pub(crate) first: bool,
    /// The last run's value, or for a seeded member its last output.
    buf: Option<Erased>,
    /// A seeded member's current value inside an episode.
    s: Option<Erased>,
    /// Every dep any run of this claim recorded, for the stale check.
    deps: Vec<(NodeId, u32)>,
    /// The deps of the last run: its reads of other members are the
    /// iteration's graph.
    last: Vec<(NodeId, u32)>,
    /// Every dep the runs since the episode's last restart recorded, in
    /// first-read order: the landed value depends on all of them, so the
    /// landing installs them all.
    installed: Vec<NodeId>,
    pub(crate) wrote: bool,
    /// The last run reached the node's child index.
    opened: bool,
    /// The landing changed the slot's value.
    pub(crate) changed: bool,
    /// The `POISONED` and `HANDLED` flags of the runs' frames.
    poison: u16,
    /// A run of this claim panicked in user code.
    pub(crate) failed: bool,
    /// A run's frame took `HOLD`, as a cycle closed on this thread's stack
    /// under it: the landing is held, as that frame's pop would hold it.
    pub(crate) hold: bool,
    /// `Cycle(M)` while the node is a member of a running episode.
    pub(crate) episode: Option<Cycle>,
    /// That episode's id, 0 when there is none.
    pub(crate) ep: u64,
    /// The member is in the episode's set of unseeded members that read
    /// each other.
    pub(crate) in_cu: bool,
    /// None until speculation writes, reads or sets a value of the claim,
    /// so a thread without a budget checks one pointer.
    spec: Option<Box<Spec>>,
}

/// What speculation left on a claim.
#[derive(Default)]
struct Spec {
    /// While the buffer and run record come from a run made under
    /// speculation: the record from before that run's write.
    pre: Option<RunRecord>,
    /// 0 until a speculative run's read of the buffer notes it, and 0
    /// again once the buffer is written or taken, so a noted stamp tells
    /// whether the buffer read is still the one held.
    stamp: u64,
    /// `stamp`'s counterpart for `s`.
    s_stamp: u64,
    /// `s` was set under speculation.
    s_spec: bool,
}

impl Held {
    /// Hands the claim, before it enters the held map, to the frame at
    /// `owner`, of `node` (`u32::MAX` for whichever frame is there),
    /// virtual at a path or not. Inside the map, [`HeldMap::own`] does.
    pub(crate) fn own(&mut self, owner: u32, node: u32, virtual_frame: Option<u64>) {
        debug_assert_eq!(self.group, 0, "incr-core: the claim is out of the held map");
        self.owner = Owner::new(owner, node, virtual_frame);
    }

    pub(crate) fn anchor(&self) -> u32 {
        self.anchor
    }

    pub(crate) fn new(generation: u32, anchor: u32, seeded: bool, first: bool) -> Self {
        Self {
            generation,
            anchor,
            owner: Owner::new(anchor, u32::MAX, None),
            group: 0,
            group_ix: 0,
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
            poison: 0,
            failed: false,
            hold: false,
            episode: None,
            ep: 0,
            in_cu: false,
            spec: None,
        }
    }

    /// Records a finished run: its deps, buffer, whether it reached the
    /// child index and its frame's `POISONED` and `HANDLED` flags
    /// (`poison`). Under speculation it first
    /// keeps the record it replaces, and no look adopts the run; outside,
    /// it settles first. Returns
    /// the buffer it replaces and the value a settle dropped, for the
    /// caller to drop outside the held map.
    pub(crate) fn record(
        &mut self,
        look: Look,
        deps: Vec<(NodeId, u32)>,
        buf: Option<Erased>,
        opened: bool,
        poison: u16,
    ) -> (Option<Erased>, Option<Unbuilt>) {
        let spec_buf = if look.speculative {
            self.keep_record().basis = None;
            None
        } else {
            self.settle()
        };
        self.unmark_buf();
        for &d in &deps {
            if !self.deps.contains(&d) {
                self.deps.push(d);
            }
            if !self.installed.contains(&d.0) {
                self.installed.push(d.0);
            }
        }
        self.last = deps;
        self.opened = opened;
        self.poison |= poison;
        (std::mem::replace(&mut self.buf, buf), spec_buf)
    }

    /// [`Self::record`] of a speculative run that read only real state,
    /// `basis`. Every speculative run since the first is merged into the
    /// kept record, so an adoption rests on all of their reads.
    #[cold]
    #[inline(never)]
    pub(crate) fn record_with_basis(
        &mut self,
        deps: Vec<(NodeId, u32)>,
        buf: Option<Erased>,
        opened: bool,
        poison: u16,
        basis: Basis,
    ) -> (Option<Erased>, Option<Unbuilt>) {
        let earlier = self.pre_spec_mut().map(|r| r.basis.take());
        let out = self.record(Look::SPECULATIVE, deps, buf, opened, poison);
        self.keep_record().basis = match earlier {
            None => Some(basis),
            Some(Some(mut kept)) => {
                kept.held.extend(basis.held);
                kept.clean.extend(basis.clean);
                Some(kept)
            }
            Some(None) => None,
        };
        out
    }

    /// The claim's run as `look` sees it, and the value a settle dropped,
    /// for the caller to drop outside the held map.
    fn look(&mut self, look: Look) -> (Seen<&mut Held>, Option<Unbuilt>) {
        let spec_buf = self.settle_for(look);
        (Seen(self), spec_buf)
    }

    /// [`Self::look`] at a claim out of the held map.
    pub(crate) fn into_look(mut self, look: Look) -> (Seen<Held>, Option<Unbuilt>) {
        let spec_buf = self.settle_for(look);
        (Seen(self), spec_buf)
    }

    /// A seeded member's current value inside an episode.
    pub(crate) fn s(&self) -> &Option<Erased> {
        &self.s
    }

    /// Replaces the current value, set under `look`, returning the one it
    /// replaces.
    #[inline]
    pub(crate) fn replace_s(&mut self, s: Option<Erased>, look: Look) -> Option<Erased> {
        if let Some(m) = &mut self.spec {
            m.s_stamp = 0;
            m.s_spec = false;
        }
        if look.speculative && s.is_some() {
            self.spec().s_spec = true;
        }
        std::mem::replace(&mut self.s, s)
    }

    #[inline]
    pub(crate) fn take_s(&mut self) -> Option<Erased> {
        if let Some(m) = &mut self.spec {
            m.s_stamp = 0;
            m.s_spec = false;
        }
        self.s.take()
    }

    /// For a change to how the claim reads (its episode, or whether it is
    /// one of the members that read each other): a buffer read before is
    /// no longer what a read now gives.
    #[inline]
    pub(crate) fn renew_stamp(&mut self) {
        self.unmark_buf();
    }

    #[inline]
    fn unmark_buf(&mut self) {
        if let Some(m) = &mut self.spec {
            m.stamp = 0;
        }
    }

    fn spec(&mut self) -> &mut Spec {
        self.spec.get_or_insert_with(Box::default)
    }

    fn pre_spec_mut(&mut self) -> Option<&mut RunRecord> {
        self.spec.as_mut()?.pre.as_mut()
    }

    /// The buffer's noted stamp, 0 if none is.
    fn buf_stamp(&self) -> u64 {
        self.spec.as_ref().map_or(0, |m| m.stamp)
    }

    /// Takes the buffer to drop it, leaving the run record as it is.
    pub(crate) fn discard_buf(&mut self) -> Option<Discarded> {
        self.unmark_buf();
        self.buf.take().map(|buf| Discarded {
            _values: (Some(buf), None),
        })
    }

    /// The buffer and seeded value to drop, for a claim released without a
    /// landing that reads its run.
    pub(crate) fn into_values(self) -> Discarded {
        Discarded {
            _values: (self.buf, self.s),
        }
    }

    /// Whether the buffer and record are a speculative run's.
    pub(crate) fn spec_written(&self) -> bool {
        self.spec.as_ref().is_some_and(|m| m.pre.is_some())
    }

    /// Keeps a speculative run's buffer and record as the claim's own, for
    /// a landing that speculation may make.
    pub(crate) fn adopt(&mut self) {
        if let Some(m) = &mut self.spec {
            m.pre = None;
        }
    }

    fn basis(&self) -> Option<&Basis> {
        self.spec.as_ref()?.pre.as_ref()?.basis.as_ref()
    }

    /// For a speculative run proven not real: no later look adopts it.
    fn drop_basis(&mut self) {
        if let Some(r) = self.pre_spec_mut() {
            r.basis = None;
        }
    }

    /// Before a run's buffer and record are written under speculation:
    /// keeps the record they replace, unless an earlier speculative run's
    /// write already did.
    fn keep_record(&mut self) -> &mut RunRecord {
        let spec = self.spec.get_or_insert_with(Box::default);
        spec.pre.get_or_insert_with(|| RunRecord {
            deps: self.deps.clone(),
            last: self.last.clone(),
            installed: self.installed.clone(),
            opened: self.opened,
            poison: self.poison,
            basis: None,
        })
    }

    fn settle_for(&mut self, look: Look) -> Option<Unbuilt> {
        if look.speculative {
            None
        } else {
            self.settle()
        }
    }

    /// Puts back the record a speculative run replaced and returns the
    /// value it buffered, as the unbounded run may never make that run.
    #[inline]
    fn settle(&mut self) -> Option<Unbuilt> {
        self.spec.as_ref()?;
        self.settle_spec()
    }

    #[cold]
    #[inline(never)]
    fn settle_spec(&mut self) -> Option<Unbuilt> {
        let spec = self.spec.as_mut()?;
        let r = spec.pre.take()?;
        spec.stamp = 0;
        self.deps = r.deps;
        self.last = r.last;
        self.installed = r.installed;
        self.opened = r.opened;
        self.poison = r.poison;
        self.buf
            .take()
            .map(|buf| Unbuilt::<dyn std::any::Any>::new(buf))
    }
}

/// How a look at a claim's run stands to speculation: one made under it
/// sees what a speculative run left, one made outside it settles the
/// claim first. Only the thread's speculation and a driver entry's or
/// virtual frame's own flag make one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Look {
    speculative: bool,
}

impl Look {
    const SPECULATIVE: Self = Self { speculative: true };
    #[cfg(all(test, not(loom)))]
    const REAL: Self = Self { speculative: false };

    pub(crate) fn is_speculative(self) -> bool {
        self.speculative
    }

    /// A look from this thread's run.
    pub(crate) fn here() -> Self {
        Self::here_or(false)
    }

    /// [`Self::here`], speculative also when `spec`: a driver entry's or
    /// virtual frame's own check of its node.
    pub(crate) fn here_or(spec: bool) -> Self {
        Self {
            speculative: spec || replay::speculative(),
        }
    }
}

/// A claim whose run a [`Look`] has seen: the only way to read its buffer
/// and run record. [`Held::discard_buf`] and [`Held::into_values`] only hand
/// them over to be dropped.
pub(crate) struct Seen<H>(H);

impl<H: Borrow<Held>> Deref for Seen<H> {
    type Target = Held;

    fn deref(&self) -> &Held {
        self.0.borrow()
    }
}

impl<H: BorrowMut<Held>> DerefMut for Seen<H> {
    fn deref_mut(&mut self) -> &mut Held {
        self.0.borrow_mut()
    }
}

impl<H: Borrow<Held>> Seen<H> {
    pub(crate) fn buf(&self) -> &Option<Erased> {
        &self.0.borrow().buf
    }

    pub(crate) fn deps(&self) -> &[(NodeId, u32)] {
        &self.0.borrow().deps
    }

    pub(crate) fn last(&self) -> &[(NodeId, u32)] {
        &self.0.borrow().last
    }

    pub(crate) fn installed(&self) -> &[NodeId] {
        &self.0.borrow().installed
    }

    pub(crate) fn opened(&self) -> bool {
        self.0.borrow().opened
    }

    /// The runs' `POISONED` and `HANDLED` flags: either lands Dirty.
    pub(crate) fn poison(&self) -> u16 {
        self.0.borrow().poison
    }
}

impl<H: BorrowMut<Held>> Seen<H> {
    pub(crate) fn take_buf(&mut self) -> Option<Erased> {
        let e = self.0.borrow_mut();
        e.unmark_buf();
        e.buf.take()
    }

    pub(crate) fn take_deps(&mut self) -> Vec<(NodeId, u32)> {
        std::mem::take(&mut self.0.borrow_mut().deps)
    }

    pub(crate) fn clear_installed(&mut self) {
        self.0.borrow_mut().installed.clear();
    }
}

impl Seen<Held> {
    pub(crate) fn into_values(self) -> Discarded {
        self.0.into_values()
    }

    pub(crate) fn into_held(self) -> Held {
        self.0
    }
}

/// Values taken off a claim only to be dropped.
pub(crate) struct Discarded {
    _values: (Option<Erased>, Option<Erased>),
}

/// What a speculative run read, for [`HeldMap::look`] to check: the held
/// members it read, each with what it read, and the other nodes, each
/// Clean when the run recorded, with its generation and `changed_at`.
#[derive(Debug, Default)]
pub(crate) struct Basis {
    pub(crate) held: Vec<(u32, MemberRead)>,
    pub(crate) clean: Vec<(NodeId, u32, u64)>,
}

/// A speculative run's read of a held member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MemberRead {
    /// The member's buffer, by its stamp.
    Buffer(u64),
    /// The member's episode's cycle, as an unseeded member that reads
    /// another such member gets it.
    Cycle(Cycle),
    /// A strategy member's current value, by its stamp.
    Current(u64),
}

static NEXT_STAMP: AtomicU64 = AtomicU64::new(1);

#[cfg(test)]
thread_local! {
    pub(crate) static ADOPTED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// `stamp`, given a value no other note took if it has none.
fn noted(stamp: &mut u64) -> u64 {
    if *stamp == 0 {
        *stamp = NEXT_STAMP.fetch_add(1, Ordering::Relaxed);
    }
    *stamp
}

/// A claim's run record, kept while a speculative run's is written over it.
struct RunRecord {
    deps: Vec<(NodeId, u32)>,
    last: Vec<(NodeId, u32)>,
    installed: Vec<NodeId>,
    opened: bool,
    poison: u16,
    /// What the speculative runs since read, if they read only real state:
    /// a look outside speculation adopts them while all of it is unchanged.
    basis: Option<Basis>,
}

/// An episode between member runs. A doom that discards a member run
/// suspends it in a replay context until that run's virtual frame pops.
pub(crate) struct Episode {
    pub(crate) id: u64,
    pub(crate) root: NodeId,
    pub(crate) cidx: u32,
    pub(crate) validated: bool,
    /// The landing's one cut to the root's SCC in the final reads has run,
    /// or is skipped (`uncut`).
    pub(crate) pruned: bool,
    pub(crate) members: Vec<u32>,
    pub(crate) cu: Vec<u32>,
    /// The iteration state since the last restart, `None` at a restart.
    pub(crate) round: Option<Round>,
    /// The root's compute is a walk's entry, which contains the episode's
    /// abort.
    pub(crate) walked: bool,
    /// The episode skipped the cut to the root's SCC: the root dissolved
    /// an episode earlier in this entry, or the landing's one cut shrank
    /// it and it ran again uncut.
    pub(crate) uncut: bool,
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

/// The held map, indexed by anchor and by owner position: the queries
/// every driver step makes ask for the claims at or above a position, and
/// a scan of the whole map per step makes a long cycle quadratic. Claims
/// that share an owner share an owner group, so that a pop handing every
/// claim its subtree deferred to its parent merges groups instead of
/// moving each claim.
pub(crate) struct HeldMap {
    claims: HashMap<(RuntimeId, u32), Held, FxBuildHasher>,
    by_anchor: BTreeSet<(RuntimeId, u32, u32)>,
    groups: HashMap<u64, Group, FxBuildHasher>,
    /// `(rt, owner position, group)` of every group.
    by_owner: BTreeSet<(RuntimeId, u32, u64)>,
    /// The groups a virtual frame owns.
    by_virtual_owner: BTreeSet<(RuntimeId, u32, u64)>,
    next_group: u64,
    /// `(rt, reader position, member, read)` of each read of a held member
    /// a speculative run made, until that run records.
    spec_reads: Vec<(RuntimeId, u32, u32, MemberRead)>,
    /// Some claim may hold a speculative run's record; false again once the
    /// map is empty, so a thread that never speculates looks up once.
    spec_written: bool,
}

struct Group {
    rt: RuntimeId,
    owner: Owner,
    members: Vec<u32>,
}

impl std::ops::Index<&(RuntimeId, u32)> for HeldMap {
    type Output = Held;

    fn index(&self, key: &(RuntimeId, u32)) -> &Held {
        &self.claims[key]
    }
}

impl HeldMap {
    fn new() -> Self {
        Self {
            claims: HashMap::default(),
            by_anchor: BTreeSet::new(),
            groups: HashMap::default(),
            by_owner: BTreeSet::new(),
            by_virtual_owner: BTreeSet::new(),
            next_group: 1,
            spec_reads: Vec::new(),
            spec_written: false,
        }
    }

    fn span<T: Ord + Copy>(
        rt: RuntimeId,
        from: u32,
        to: u32,
        min: T,
        max: T,
    ) -> std::ops::RangeInclusive<(RuntimeId, u32, T)> {
        (rt, from, min)..=(rt, to, max)
    }

    fn index_group(&mut self, g: u64) {
        let group = &self.groups[&g];
        let key = (group.rt, group.owner.pos, g);
        self.by_owner.insert(key);
        if group.owner.virtual_frame.is_some() {
            self.by_virtual_owner.insert(key);
        }
    }

    fn unindex_group(&mut self, g: u64) {
        let group = &self.groups[&g];
        let key = (group.rt, group.owner.pos, g);
        self.by_owner.remove(&key);
        self.by_virtual_owner.remove(&key);
    }

    fn new_group(&mut self, rt: RuntimeId, owner: Owner) -> u64 {
        let g = self.next_group;
        self.next_group += 1;
        self.groups.insert(
            g,
            Group {
                rt,
                owner,
                members: Vec::new(),
            },
        );
        self.index_group(g);
        g
    }

    /// Adds `key`'s claim, out of every group, to group `g`.
    fn join(&mut self, (rt, slot): (RuntimeId, u32), g: u64) {
        let members = &mut self
            .groups
            .get_mut(&g)
            .expect("incr-core: the group exists")
            .members;
        let e = self
            .claims
            .get_mut(&(rt, slot))
            .expect("incr-core: the claim is held");
        e.group = g;
        e.group_ix = members.len();
        members.push(slot);
    }

    /// Takes `key`'s claim out of its group, returning the group's owner if
    /// the claim was its last member.
    fn leave(&mut self, (rt, slot): (RuntimeId, u32)) -> Option<Owner> {
        let e = self
            .claims
            .get_mut(&(rt, slot))
            .expect("incr-core: the claim is held");
        let (g, ix) = (e.group, e.group_ix);
        e.group = 0;
        let group = self
            .groups
            .get_mut(&g)
            .expect("incr-core: the group exists");
        group.members.swap_remove(ix);
        if let Some(&moved) = group.members.get(ix) {
            self.claims
                .get_mut(&(rt, moved))
                .expect("incr-core: a member is held")
                .group_ix = ix;
            return None;
        }
        if !group.members.is_empty() {
            return None;
        }
        self.unindex_group(g);
        self.groups.remove(&g).map(|group| group.owner)
    }

    pub(crate) fn insert(&mut self, key: (RuntimeId, u32), mut held: Held) -> Option<Held> {
        let displaced = self.remove(&key);
        self.by_anchor.insert((key.0, held.anchor, key.1));
        let g = self.new_group(key.0, std::mem::take(&mut held.owner));
        self.spec_written |= held.spec_written();
        self.claims.insert(key, held);
        self.join(key, g);
        displaced
    }

    pub(crate) fn remove(&mut self, key: &(RuntimeId, u32)) -> Option<Held> {
        let g = self.claims.get(key)?.group;
        let owner = match self.leave(*key) {
            Some(owner) => owner,
            None => self.groups[&g].owner.clone(),
        };
        let mut e = self
            .claims
            .remove(key)
            .expect("incr-core: the claim is held");
        self.by_anchor.remove(&(key.0, e.anchor, key.1));
        e.owner = owner;
        if self.claims.is_empty() {
            self.spec_written = false;
        }
        Some(e)
    }

    pub(crate) fn get(&self, key: &(RuntimeId, u32)) -> Option<&Held> {
        self.claims.get(key)
    }

    pub(crate) fn get_mut(&mut self, key: &(RuntimeId, u32)) -> Option<&mut Held> {
        self.claims.get_mut(key)
    }

    /// Claim `key`'s run as `look` sees it, and the value a settle dropped.
    /// A look outside speculation adopts a speculative run, with the
    /// speculative runs whose buffers it read, while everything they read
    /// is unchanged ([`Self::adoptable`]); otherwise it settles the claim.
    /// `clean` gives a node's generation and `changed_at` while it is
    /// Clean.
    #[inline(always)]
    pub(crate) fn look(
        &mut self,
        key: &(RuntimeId, u32),
        look: Look,
        clean: &impl Fn(NodeId) -> Option<(u32, u64)>,
    ) -> Option<(Seen<&mut Held>, Option<Unbuilt>)> {
        if self.spec_written && !look.speculative {
            self.adopt_if_real(key, clean);
        }
        self.claims.get_mut(key).map(|e| e.look(look))
    }

    #[cold]
    #[inline(never)]
    fn adopt_if_real(
        &mut self,
        key: &(RuntimeId, u32),
        clean: &dyn Fn(NodeId) -> Option<(u32, u64)>,
    ) {
        if !self.claims.get(key).is_some_and(Held::spec_written) {
            return;
        }
        let mut verdicts = HashMap::default();
        let adopted = self.adoptable(key.0, key.1, clean, &mut verdicts);
        for (m, real) in verdicts {
            if let Some(e) = self.claims.get_mut(&(key.0, m)) {
                if adopted && real {
                    #[cfg(test)]
                    ADOPTED.with(|a| a.borrow_mut().push(m));
                    e.adopt();
                } else if !real {
                    e.drop_basis();
                }
            }
        }
    }

    /// Whether claim `slot`'s buffer is real: a run's outside speculation,
    /// or a speculative run's whose reads were all of real state and are
    /// unchanged, each Clean node at its generation and `changed_at`, each
    /// member's buffer the one read and itself real, and each member read
    /// as its episode's cycle still in that episode as such a member.
    /// `verdicts` keeps each claim's answer. Buffer reads form no cycle: a
    /// record resets the claim's stamp after its reads, so of two claims
    /// each reading the other's current buffer one would have recorded
    /// after the other's read.
    fn adoptable(
        &self,
        rt: RuntimeId,
        slot: u32,
        clean: &dyn Fn(NodeId) -> Option<(u32, u64)>,
        verdicts: &mut HashMap<u32, bool, FxBuildHasher>,
    ) -> bool {
        if let Some(&real) = verdicts.get(&slot) {
            return real;
        }
        verdicts.insert(slot, true);
        let real = self.reads_real(rt, slot, clean, verdicts);
        verdicts.insert(slot, real);
        real
    }

    fn reads_real(
        &self,
        rt: RuntimeId,
        slot: u32,
        clean: &dyn Fn(NodeId) -> Option<(u32, u64)>,
        verdicts: &mut HashMap<u32, bool, FxBuildHasher>,
    ) -> bool {
        let Some(e) = self.claims.get(&(rt, slot)) else {
            return false;
        };
        if !e.spec_written() {
            return true;
        }
        let Some(basis) = e.basis() else {
            return false;
        };
        basis
            .clean
            .iter()
            .all(|&(d, generation, at)| clean(d) == Some((generation, at)))
            && basis.held.iter().all(|(m, read)| match read {
                MemberRead::Buffer(stamp) => {
                    self.claims
                        .get(&(rt, *m))
                        .is_some_and(|r| r.buf_stamp() == *stamp)
                        && self.adoptable(rt, *m, clean, verdicts)
                }
                MemberRead::Current(stamp) => self
                    .claims
                    .get(&(rt, *m))
                    .and_then(|r| r.spec.as_ref())
                    .is_some_and(|m| m.s_stamp == *stamp && !m.s_spec),
                MemberRead::Cycle(cycle) => self
                    .claims
                    .get(&(rt, *m))
                    .is_some_and(|r| r.in_cu && r.episode.as_ref() == Some(cycle)),
            })
    }

    /// Claim `key`, if held, for a [`Held::record`] under `look`, which
    /// must go through here so that the map knows of speculative records.
    #[inline(always)]
    pub(crate) fn recording(&mut self, key: &(RuntimeId, u32), look: Look) -> Option<&mut Held> {
        self.spec_written |= look.speculative;
        self.claims.get_mut(key)
    }

    /// Drops the reads noted at position `at` or above, of a run that
    /// keeps no basis.
    #[inline]
    pub(crate) fn drop_spec_reads(&mut self, rt: RuntimeId, at: u32) {
        if !self.spec_reads.is_empty() {
            self.drop_spec_reads_from(rt, at);
        }
    }

    #[cold]
    #[inline(never)]
    fn drop_spec_reads_from(&mut self, rt: RuntimeId, at: u32) {
        self.spec_reads.retain(|(r, c, ..)| *r != rt || *c < at);
        self.mark_spec_reads();
    }

    fn push_spec_read(&mut self, read: (RuntimeId, u32, u32, MemberRead)) {
        self.spec_reads.push(read);
        self.mark_spec_reads();
    }

    fn mark_spec_reads(&self) {
        value_code::set_bits(value_code::SPEC_READS_BIT, !self.spec_reads.is_empty());
    }

    /// Notes a speculative run's read, at position `at`, of member `slot`'s
    /// buffer.
    pub(crate) fn note_spec_read(&mut self, rt: RuntimeId, at: u32, slot: u32) {
        if let Some(e) = self.claims.get_mut(&(rt, slot)) {
            let read = MemberRead::Buffer(noted(&mut e.spec().stamp));
            self.push_spec_read((rt, at, slot, read));
        }
    }

    /// Notes a speculative run's read, at position `at`, of strategy member
    /// `slot`'s current value.
    pub(crate) fn note_spec_current_read(&mut self, rt: RuntimeId, at: u32, slot: u32) {
        if let Some(e) = self.claims.get_mut(&(rt, slot)) {
            let read = MemberRead::Current(noted(&mut e.spec().s_stamp));
            self.push_spec_read((rt, at, slot, read));
        }
    }

    /// Notes a speculative run's read, at position `at`, of member `slot`
    /// as `cycle`, its episode's.
    pub(crate) fn note_spec_cycle_read(&mut self, rt: RuntimeId, at: u32, slot: u32, cycle: Cycle) {
        self.push_spec_read((rt, at, slot, MemberRead::Cycle(cycle)));
    }

    /// The basis of a speculative run at position `at` that read `deps`,
    /// if each was real state: a member read as noted, or a node `clean`
    /// finds Clean at the generation read. Drops every read noted at `at`
    /// or above.
    pub(crate) fn take_basis(
        &mut self,
        rt: RuntimeId,
        at: u32,
        deps: &[(NodeId, u32)],
        clean: &dyn Fn(NodeId) -> Option<(u32, u64)>,
    ) -> Option<Basis> {
        let mut held = Vec::new();
        let mut reads = std::mem::take(&mut self.spec_reads);
        reads.retain(|(r, c, m, read)| {
            let mine = *r == rt && *c >= at;
            if mine && *c == at {
                held.push((*m, read.clone()));
            }
            !mine
        });
        self.spec_reads = reads;
        self.mark_spec_reads();
        let mut basis = Basis {
            held,
            clean: Vec::new(),
        };
        if basis
            .held
            .iter()
            .any(|&(m, _)| !self.claims.contains_key(&(rt, m)))
        {
            return None;
        }
        for &(d, generation) in deps {
            if self.claims.contains_key(&(rt, d.0)) {
                if !basis.held.iter().any(|&(m, _)| m == d.0) {
                    return None;
                }
            } else {
                let (g, at) = clean(d).filter(|&(g, _)| g == generation)?;
                basis.clean.push((d, g, at));
            }
        }
        Some(basis)
    }

    pub(crate) fn contains_key(&self, key: &(RuntimeId, u32)) -> bool {
        self.claims.contains_key(key)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Moves `key`'s claim to anchor `anchor`.
    pub(crate) fn set_anchor(&mut self, key: (RuntimeId, u32), anchor: u32) {
        let e = self
            .claims
            .get_mut(&key)
            .expect("incr-core: the claim is held");
        if e.anchor != anchor {
            self.by_anchor.remove(&(key.0, e.anchor, key.1));
            self.by_anchor.insert((key.0, anchor, key.1));
            e.anchor = anchor;
        }
    }

    /// The episode root at `cidx` holds `rt`'s claims `slots`: anchored
    /// and owned there, by whichever frame runs at that position.
    pub(crate) fn root_at(&mut self, rt: RuntimeId, slots: &[u32], cidx: u32) {
        for &m in slots {
            self.set_anchor((rt, m), cidx);
        }
        self.hand_over(rt, &[], slots, cidx, u32::MAX, None);
    }

    /// [`Held::own`] for a claim in the map.
    pub(crate) fn own(
        &mut self,
        key: (RuntimeId, u32),
        owner: u32,
        node: u32,
        virtual_frame: Option<u64>,
    ) {
        self.hand_over(key.0, &[], &[key.1], owner, node, virtual_frame);
    }

    /// Hands every claim of `rt`'s owner groups `groups`, and `rt`'s claims
    /// `slots`, to the frame at `owner`, of `node`, virtual at a path or
    /// not, as [`Held::own`] would each. The largest of `groups` takes the
    /// rest, so that a claim moves a logarithmic number of times however
    /// often its owner is handed down.
    pub(crate) fn hand_over(
        &mut self,
        rt: RuntimeId,
        groups: &[u64],
        slots: &[u32],
        owner: u32,
        node: u32,
        virtual_frame: Option<u64>,
    ) {
        if groups.is_empty() && slots.is_empty() {
            return;
        }
        let owner = Owner::new(owner, node, virtual_frame);
        let target = groups
            .iter()
            .copied()
            .max_by_key(|g| self.groups[g].members.len());
        let target = match target {
            Some(g) => {
                self.unindex_group(g);
                self.groups
                    .get_mut(&g)
                    .expect("incr-core: the group exists")
                    .owner = owner;
                self.index_group(g);
                g
            }
            None => self.new_group(rt, owner),
        };
        for &g in groups.iter().filter(|&&g| g != target) {
            self.unindex_group(g);
            let group = self.groups.remove(&g).expect("incr-core: the group exists");
            for m in group.members {
                self.join((rt, m), target);
            }
        }
        for &m in slots {
            if self.claims[&(rt, m)].group != target {
                self.leave((rt, m));
                self.join((rt, m), target);
            }
        }
    }

    /// The frame that retries virtual owner group `g`'s frame owns its
    /// claims from now, at the same position, of the same node.
    pub(crate) fn own_retry(&mut self, g: u64) {
        self.unindex_group(g);
        self.groups
            .get_mut(&g)
            .expect("incr-core: the group exists")
            .owner
            .virtual_frame = None;
        self.index_group(g);
    }

    /// Records that the virtual frame at `cidx`, at `path`, popped, in each
    /// owner group a virtual frame above it is.
    pub(crate) fn note_ancestor(&mut self, rt: RuntimeId, cidx: u32, path: u64) {
        let Some(from) = cidx.checked_add(1) else {
            return;
        };
        for &(_, _, g) in self
            .by_virtual_owner
            .range(Self::span(rt, from, u32::MAX, 0, u64::MAX))
        {
            if let Some(group) = self.groups.get_mut(&g) {
                group.owner.ancestry.push((cidx, path));
            }
        }
    }

    /// The slots of `rt`'s claims anchored at or above `from`, lowest
    /// anchor first.
    pub(crate) fn anchored(&self, rt: RuntimeId, from: u32) -> impl Iterator<Item = u32> + '_ {
        self.by_anchor
            .range(Self::span(rt, from, u32::MAX, 0, u32::MAX))
            .map(|&(_, _, slot)| slot)
    }

    /// [`Self::anchored`], collected, for a caller that changes the map.
    pub(crate) fn anchored_from(&self, rt: RuntimeId, from: u32) -> Vec<u32> {
        self.anchored(rt, from).collect()
    }

    /// `rt`'s owner groups at positions `from..=to`.
    pub(crate) fn owners(&self, rt: RuntimeId, from: u32, to: u32) -> Vec<u64> {
        self.by_owner
            .range(Self::span(rt, from, to, 0, u64::MAX))
            .map(|&(_, _, g)| g)
            .collect()
    }

    /// `rt`'s virtual owner groups at positions `from..=to`.
    pub(crate) fn virtual_owners(&self, rt: RuntimeId, from: u32, to: u32) -> Vec<u64> {
        self.by_virtual_owner
            .range(Self::span(rt, from, to, 0, u64::MAX))
            .map(|&(_, _, g)| g)
            .collect()
    }

    /// Whether a virtual frame at or above `from` owns a claim of `rt`.
    pub(crate) fn any_virtually_owned_from(&self, rt: RuntimeId, from: u32) -> bool {
        self.by_virtual_owner
            .range(Self::span(rt, from, u32::MAX, 0, u64::MAX))
            .next()
            .is_some()
    }

    /// Whether a frame at or above `from` owns a claim of `rt`.
    pub(crate) fn any_owned_from(&self, rt: RuntimeId, from: u32) -> bool {
        self.by_owner
            .range(Self::span(rt, from, u32::MAX, 0, u64::MAX))
            .next()
            .is_some()
    }

    pub(crate) fn owner(&self, g: u64) -> &Owner {
        &self.groups[&g].owner
    }

    pub(crate) fn members(&self, g: u64) -> &[u32] {
        &self.groups[&g].members
    }
}

/// The engine entry state of a thread, in one block so that an entry
/// and its exit touch one thread-local.
struct EntryState {
    depth: Cell<u32>,
    /// The key of the runtime whose entry is the innermost one: its id,
    /// with [`SHARED_KEY`] for a `Shared` runtime.
    cur_rt: Cell<u64>,
    /// This thread's wound-wait ticket, 0 until its first registration
    /// inside the current outermost entry, which keeps it across retries.
    ticket: Cell<u64>,
    /// Set by a wound until the outermost entry retries.
    wounded: Cell<bool>,
    /// A take-over met a claim the outermost entry keeps unreserved
    /// across its retries, as long as the ticket.
    taken: Cell<bool>,
    /// Open `KeptUnwind`s: while any is, the attempt's flight belongs to
    /// the unwind whose kind it keeps.
    kept: Cell<u32>,
    /// The control block of the doomed frame that took its first
    /// placeholder and runs on in user code, or 0.
    lazy: Cell<usize>,
    failures: Cell<u64>,
    /// Set when `Outer` may hold something, so that an outermost entry
    /// with nothing to reset touches only this block, which has no
    /// destructor and so no liveness check.
    outer_set: Cell<bool>,
    /// The held map has entries: a check of a thread that holds nothing
    /// reads this block and not the map, whose access tests liveness.
    held_any: Cell<bool>,
    /// A read in the outermost entry's attempt took a cycle and went on.
    /// Set only with `outer_set`.
    cycle_caught: Cell<bool>,
    /// The address of a local of the outermost entry, from which a read
    /// measures the stack it has used against a byte budget.
    stack_base: Cell<usize>,
    /// `BUDGET_CLOCK` when the outermost entry began: a runtime whose
    /// budget changed later may have frames on the stack that read under
    /// the budget before.
    budget_clock: Cell<u64>,
    /// The length of `Outer::failed_here`, read by every Shared walk.
    failed_here: Cell<usize>,
    /// This thread's [`thread_key`], 0 until first asked.
    thread_key: Cell<u32>,
}

/// Per (runtime, group record address): the record, the member a verify
/// of it failed on, the depth of this thread's frame computing that
/// member, and the runtime's record change count the verify saw.
type GroupBlockers = HashMap<(RuntimeId, usize), (Arc<GroupInfo>, GroupBlocker), FxBuildHasher>;

/// A member a verify failed on, the depth of this thread's frame computing
/// it, and the runtime's record change and in-group unrun landing counts
/// the verify saw.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct GroupBlocker {
    pub(crate) member: u32,
    pub(crate) depth: usize,
    pub(crate) rec_changes: u64,
    pub(crate) unrun: u64,
}

/// What the outermost entry resets: the roots whose episodes dissolved,
/// and the cycle the last uncaught cycle panic reported.
struct Outer {
    dissolved: Vec<(RuntimeId, u32)>,
    /// The landed group of each root whose episode dissolved in this
    /// entry, held so that no later record can reuse its address.
    dissolved_groups: Vec<(RuntimeId, u32, Arc<GroupInfo>)>,
    /// Each group record whose verify failed on a member this thread's
    /// frame computes, held so that no later record can reuse its address.
    group_blockers: GroupBlockers,
    /// The group record this thread left at each slot it changed, held so
    /// that no later record can reuse its address.
    recs_left: Vec<(RuntimeId, u32, Option<Arc<GroupInfo>>)>,
    last_cycle: Option<Cycle>,
    /// Each runtime whose budget change this thread has met, with the
    /// `BUDGET_CLOCK` count of the change.
    budgets_met: Vec<(RuntimeId, u64)>,
    /// The nodes this thread has landed Failed, at their generations.
    failed_here: Vec<(RuntimeId, u32, u32)>,
}

#[cfg(not(loom))]
thread_local! {
    static HELD: RefCell<HeldMap> = RefCell::new(HeldMap::new());
    static ENTRY: EntryState = const {
        EntryState {
            depth: Cell::new(0),
            cur_rt: Cell::new(0),
            ticket: Cell::new(0),
            wounded: Cell::new(false),
            taken: Cell::new(false),
            kept: Cell::new(0),
            lazy: Cell::new(0),
            failures: Cell::new(0),
            outer_set: Cell::new(false),
            held_any: Cell::new(false),
            cycle_caught: Cell::new(false),
            stack_base: Cell::new(0),
            budget_clock: Cell::new(0),
            failed_here: Cell::new(0),
            thread_key: Cell::new(0),
        }
    };
    static OWN_EPISODES: RefCell<Vec<(RuntimeId, u64)>> = const { RefCell::new(Vec::new()) };
    static OUTER_RTS: RefCell<OuterRts> = RefCell::new(OuterRts::default());
    static OUTER: RefCell<Outer> = const {
        RefCell::new(Outer {
            dissolved: Vec::new(),
            dissolved_groups: Vec::new(),
            group_blockers: HashMap::with_hasher(FxBuildHasher),
            recs_left: Vec::new(),
            last_cycle: None,
            budgets_met: Vec::new(),
            failed_here: Vec::new(),
        })
    };
}

#[cfg(loom)]
loom::thread_local! {
    static HELD: RefCell<HeldMap> = RefCell::new(HeldMap::new());
    static ENTRY: EntryState = EntryState {
        depth: Cell::new(0),
        cur_rt: Cell::new(0),
        ticket: Cell::new(0),
        wounded: Cell::new(false),
        taken: Cell::new(false),
        kept: Cell::new(0),
        lazy: Cell::new(0),
        failures: Cell::new(0),
        outer_set: Cell::new(false),
        held_any: Cell::new(false),
        cycle_caught: Cell::new(false),
        stack_base: Cell::new(0),
        budget_clock: Cell::new(0),
        failed_here: Cell::new(0),
        thread_key: Cell::new(0),
    };
    static OWN_EPISODES: RefCell<Vec<(RuntimeId, u64)>> = RefCell::new(Vec::new());
    static OUTER_RTS: RefCell<OuterRts> = RefCell::new(OuterRts::default());
    static OUTER: RefCell<Outer> = RefCell::new(Outer {
        dissolved: Vec::new(),
        dissolved_groups: Vec::new(),
        group_blockers: HashMap::with_hasher(FxBuildHasher),
        recs_left: Vec::new(),
        last_cycle: None,
        budgets_met: Vec::new(),
        failed_here: Vec::new(),
    });
}

static NEXT_THREAD_KEY: AtomicU32 = AtomicU32::new(1);

/// A number no other thread alive holds, which tells this thread's
/// episodes in a runtime's log from other threads'.
pub(crate) fn thread_key() -> u32 {
    ENTRY.with(|e| {
        if e.thread_key.get() == 0 {
            e.thread_key
                .set(NEXT_THREAD_KEY.fetch_add(1, Ordering::Relaxed).max(1));
        }
        e.thread_key.get()
    })
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
        debug_assert_eq!(
            value_code::thread_word() & value_code::SPEC_READS_BIT != 0,
            !h.spec_reads.is_empty(),
            "incr-core: the thread word tells whether speculative reads are noted"
        );
        r
    })
}

pub(crate) fn held_any() -> bool {
    ENTRY.with(|e| e.held_any.get())
}

/// Every claim in the held map, of every runtime.
pub(crate) fn held_keys() -> Vec<(RuntimeId, u32)> {
    HELD.with(|h| h.borrow().claims.keys().copied().collect())
}

pub(crate) fn holds(rt: RuntimeId, slot: u32) -> bool {
    held_any() && HELD.with(|h| h.borrow().contains_key(&(rt, slot)))
}

/// The episode `slot`'s held claim is a member of, if it is held in one.
pub(crate) fn held_episode(rt: RuntimeId, slot: u32) -> Option<u64> {
    if !held_any() {
        return None;
    }
    HELD.with(|h| h.borrow().get(&(rt, slot)).map(|e| e.ep))
        .filter(|&ep| ep != 0)
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

/// Sets the thread's unwind kind. A discard is in flight exactly while
/// the kind is [`UnwindKind::Discard`], which the attempt word mirrors for
/// waiters; leaving it after a waiter wounded the attempt marks the thread
/// wounded, so its outermost entry retries whatever it was unwinding.
pub(crate) fn set_unwind_kind(kind: Option<UnwindKind>) {
    let was = unwind_kind();
    value_code::set_unwind_bits(UnwindKind::bits(kind));
    if flight_kept() {
        return;
    }
    let discard = Some(UnwindKind::Discard);
    if kind == discard && was != discard {
        crate::attempt::enter_flight();
    } else if was == discard && kind != discard && crate::attempt::leave_flight() {
        crate::reserve::mark_taken();
        set_wounded(true);
    }
}

/// Notes that the attempt took a cycle and reads on past it.
pub(crate) fn note_cycle_caught() {
    ENTRY.with(|e| {
        e.cycle_caught.set(true);
        e.outer_set.set(true);
    });
}

pub(crate) fn cycle_caught() -> bool {
    ENTRY.with(|e| e.cycle_caught.get())
}

/// Where the engine catches a discard: if a waiter wounded the attempt
/// while it was in flight, the discard becomes a wound, decided by the
/// same CAS that ends the flight. True if it did.
pub(crate) fn settle_discard() -> bool {
    if unwind_kind() != Some(UnwindKind::Discard)
        || flight_kept()
        || !crate::attempt::leave_flight()
    {
        return false;
    }
    value_code::set_unwind_bits(UnwindKind::bits(Some(UnwindKind::Wound)));
    crate::reserve::mark_taken();
    set_wounded(true);
    true
}

/// A waiter wounded this thread's attempt while it held landings and took
/// one over (`hold::hand_over`): the thread is wounded, its run ends as
/// the wound its entry retries from, and the claims its waiters met are
/// taken. True if this set the wound's kind; inside a [`KeptUnwind`] only
/// the kept unwind's catch may, and the entry's exit finds the wound.
pub(crate) fn settle_hold() -> bool {
    if wounded() {
        return false;
    }
    crate::reserve::mark_taken();
    set_wounded(true);
    if flight_kept() {
        return false;
    }
    value_code::set_unwind_bits(UnwindKind::bits(Some(UnwindKind::Wound)));
    true
}

/// Where the engine raises a discard it caught again, past its own catch
/// sites: in flight again until the next one settles it.
pub(crate) fn reopen_discard() {
    if unwind_kind() == Some(UnwindKind::Discard) && !flight_kept() {
        crate::attempt::reopen_flight();
    }
}

/// Inside a [`KeptUnwind`], the attempt's flight is the kept unwind's,
/// which only that unwind's own catch may end, settle or reopen: an inner
/// catch that settled a wound there would raise it out of a destructor.
fn flight_kept() -> bool {
    ENTRY.with(|e| e.kept.get() != 0)
}

/// A doomed `Shared` frame takes its first placeholder and runs on in user
/// code, its doom's flight open as an unwind's would be: a thread that
/// waits on the doom's reservations takes them over, so the frame may join
/// it. The frame's next engine call raises the discard instead.
pub(crate) fn open_lazy(ctl: usize) {
    ENTRY.with(|e| e.lazy.set(ctl));
    if !flight_kept() {
        crate::attempt::reopen_flight();
    }
}

/// Whether a doomed frame of this thread runs on past its placeholder.
#[inline]
pub(crate) fn lazy_open() -> bool {
    ENTRY.with(|e| e.lazy.get() != 0)
}

/// Whether frame `ctl` is the doomed frame that runs on past its
/// placeholder.
#[inline]
pub(crate) fn lazy_is(ctl: usize) -> bool {
    ENTRY.with(|e| e.lazy.get() == ctl)
}

/// An engine call from a doomed frame that runs on past its placeholder
/// raises the frame's discard, so the frame never reads, runs or writes
/// past its doom. Not while unwinding: that call keeps the unwind's kind.
#[inline]
pub(crate) fn raise_if_lazy() {
    if lazy_open() && !std::thread::panicking() {
        raise_lazy();
    }
}

#[cold]
#[inline(never)]
fn raise_lazy() -> ! {
    set_unwind_kind(Some(UnwindKind::Discard));
    std::panic::resume_unwind(Box::new(crate::cycle::DoomUnwind))
}

/// The end of frame `ctl`'s run, by return or by unwind. If the frame ran
/// on past its placeholder, its flight closes here, unless a discard it
/// raised is unwinding, whose catch settles it. True if a waiter wounded
/// the attempt meanwhile: the run ends as the wound its entry retries from.
pub(crate) fn end_lazy(ctl: usize) -> bool {
    if ENTRY.with(|e| e.lazy.get()) != ctl {
        return false;
    }
    ENTRY.with(|e| e.lazy.set(0));
    if unwind_kind() == Some(UnwindKind::Discard)
        || flight_kept()
        || !crate::attempt::leave_flight()
    {
        return false;
    }
    value_code::set_unwind_bits(UnwindKind::bits(Some(UnwindKind::Wound)));
    crate::reserve::mark_taken();
    set_wounded(true);
    true
}

/// An engine unwind's kind, set aside while a destructor that runs in that
/// unwind calls into the engine, and put back when the call returns or
/// unwinds. The call's own unwinds and catches start from no kind, so they
/// neither take the kept kind nor clobber it; an unwind that leaves the
/// call goes to the destructor's code, which the kept kind no longer needs.
/// A lazy doom open under the panic is kept the same way: its flight is
/// the panic's, and the call's own frames must not close it.
pub(crate) struct KeptUnwind(u64, usize);

impl KeptUnwind {
    /// Runs `f`, an engine call, under [`raise_if_lazy`]. Only an unwind
    /// with a kind or a lazy doom is kept: inside the call, reads start
    /// from none, so they keep nothing and their own kinds reach the
    /// catches inside the call. Only the cold path holds the guard: drop
    /// glue in the call's own frame would give every unwind through it a
    /// landing pad to stop at.
    #[inline(always)]
    pub(crate) fn around<R>(f: impl FnOnce() -> R) -> R {
        let bits = value_code::thread_word() & value_code::UNWIND_MASK;
        if bits != 0 || lazy_open() {
            return Self::kept(bits, f);
        }
        f()
    }

    /// [`Self::around`] for a read, which the doomed frame that runs on
    /// past its placeholder may make when `own` says the read is that
    /// frame's own: the read takes another placeholder, never computing or
    /// waiting, and raises the discard where it would. A wide doomed frame
    /// then pushes every doomed read in one run rather than one per run.
    #[inline(always)]
    pub(crate) fn around_read<R>(f: impl FnOnce() -> R, own: impl FnOnce() -> bool) -> R {
        let bits = value_code::thread_word() & value_code::UNWIND_MASK;
        if bits != 0 || lazy_open() {
            if std::thread::panicking() {
                return Self::kept(bits, f);
            }
            if lazy_open() && !own() {
                raise_lazy();
            }
        }
        f()
    }

    #[cold]
    #[inline(never)]
    fn kept<R>(bits: u64, f: impl FnOnce() -> R) -> R {
        if !std::thread::panicking() {
            raise_if_lazy();
            return f();
        }
        let _kept = Self::keep(bits);
        f()
    }

    #[cold]
    #[inline(never)]
    fn keep(bits: u64) -> Self {
        value_code::set_unwind_bits(0);
        let lazy = ENTRY.with(|e| {
            e.kept.set(e.kept.get() + 1);
            e.lazy.replace(0)
        });
        Self(bits, lazy)
    }
}

impl Drop for KeptUnwind {
    fn drop(&mut self) {
        value_code::set_unwind_bits(self.0);
        ENTRY.with(|e| {
            e.kept.set(e.kept.get() - 1);
            e.lazy.set(self.1);
        });
    }
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

/// Notes that `root`'s episode dissolved over its landed group `group`.
pub(crate) fn note_dissolved_group(rt: RuntimeId, root: u32, group: Arc<GroupInfo>) {
    OUTER.with(|o| o.borrow_mut().dissolved_groups.push((rt, root, group)));
    ENTRY.with(|e| e.outer_set.set(true));
}

/// Whether `group` is the record whose episode at `root` dissolved in
/// this outermost entry.
pub(crate) fn is_dissolved_group(rt: RuntimeId, root: u32, group: &Arc<GroupInfo>) -> bool {
    OUTER.with(|o| {
        o.borrow()
            .dissolved_groups
            .iter()
            .any(|(r, n, g)| *r == rt && *n == root && Arc::ptr_eq(g, group))
    })
}

/// Notes the blocker a verify of `group` failed on.
pub(crate) fn note_group_blocker(rt: RuntimeId, group: Arc<GroupInfo>, blocker: GroupBlocker) {
    let key = (rt, Arc::as_ptr(&group) as usize);
    OUTER.with(|o| o.borrow_mut().group_blockers.insert(key, (group, blocker)));
    ENTRY.with(|e| e.outer_set.set(true));
}

/// The blocker a verify of `group` last failed on in this outermost entry.
pub(crate) fn group_blocker(rt: RuntimeId, group: &Arc<GroupInfo>) -> Option<GroupBlocker> {
    if !group_blocker_skip() {
        return None;
    }
    OUTER.with(|o| {
        let o = o.borrow();
        if o.group_blockers.is_empty() {
            return None;
        }
        let key = (rt, Arc::as_ptr(group) as usize);
        o.group_blockers.get(&key).map(|&(_, blocker)| blocker)
    })
}

#[cfg(all(test, not(loom)))]
thread_local! {
    static GROUP_BLOCKER_SKIP: Cell<bool> = const { Cell::new(true) };
    static GROUP_BLOCKER_SKIPS: Cell<u64> = const { Cell::new(0) };
}

#[cfg(all(test, not(loom)))]
fn group_blocker_skip() -> bool {
    GROUP_BLOCKER_SKIP.with(Cell::get)
}

#[cfg(not(all(test, not(loom))))]
fn group_blocker_skip() -> bool {
    true
}

/// Turns this thread's blocked-record skip on or off, for tests that
/// compare a read with and without it.
#[cfg(all(test, not(loom)))]
pub(crate) fn set_group_blocker_skip(on: bool) {
    GROUP_BLOCKER_SKIP.with(|s| s.set(on));
}

#[cfg(all(test, not(loom)))]
pub(crate) fn note_group_blocker_skip() {
    GROUP_BLOCKER_SKIPS.with(|n| n.set(n.get() + 1));
}

/// The group verifies this thread returned early on a blocked record.
#[cfg(all(test, not(loom)))]
pub(crate) fn group_blocker_skips() -> u64 {
    GROUP_BLOCKER_SKIPS.with(Cell::get)
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

/// Counts budget changes across runtimes, so an outermost entry and a
/// runtime's last change can be ordered.
static BUDGET_CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A new `BUDGET_CLOCK` count, for a budget change.
pub(crate) fn tick_budget_clock() -> u64 {
    BUDGET_CLOCK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

/// Whether runtime `rt`'s budget change at `changed` is one this thread
/// meets for the first time inside its outermost entry, which notes it.
pub(crate) fn meet_budget_change(rt: RuntimeId, changed: u64) -> bool {
    if changed <= ENTRY.with(|e| e.budget_clock.get()) {
        return false;
    }
    OUTER.with(|o| {
        let met = &mut o.borrow_mut().budgets_met;
        match met.iter_mut().find(|(r, _)| *r == rt) {
            Some((_, at)) if *at >= changed => return false,
            Some((_, at)) => *at = changed,
            None => met.push((rt, changed)),
        }
        ENTRY.with(|e| e.outer_set.set(true));
        true
    })
}

pub(crate) fn note_failed_here(rt: RuntimeId, slot: u32, generation: u32) {
    OUTER.with(|o| o.borrow_mut().failed_here.push((rt, slot, generation)));
    ENTRY.with(|e| {
        e.outer_set.set(true);
        e.failed_here.set(e.failed_here.get() + 1);
    });
}

/// How many Failed landings this thread has made in its outermost entry.
pub(crate) fn failed_here_count() -> usize {
    ENTRY.with(|e| e.failed_here.get())
}

/// Whether one of this thread's Failed landings after its first `from`
/// in its outermost entry is of `slot` of `rt` at `generation`.
pub(crate) fn failed_here_since(from: usize, rt: RuntimeId, slot: u32, generation: u32) -> bool {
    OUTER.with(|o| {
        let o = o.borrow();
        o.failed_here
            .get(from..)
            .is_some_and(|f| f.contains(&(rt, slot, generation)))
    })
}

/// The stack address the outermost engine entry recorded.
pub(crate) fn stack_base() -> usize {
    ENTRY.with(|e| e.stack_base.get())
}

/// The thread's engine entry depth: 0 outside every engine entry.
pub(crate) fn depth() -> u32 {
    ENTRY.with(|e| e.depth.get())
}

/// Adds `by` to this thread's share of `rt`'s episode count, for the last
/// `OWN_RUNTIMES` runtimes it counted on. A count dropped or restarted
/// only makes a walk's own share differ from all landings, which re-plans.
pub(crate) fn own_episode(rt: RuntimeId, by: u64) {
    OWN_EPISODES.with(|o| {
        let mut o = o.borrow_mut();
        match o.iter_mut().find(|(r, _)| *r == rt) {
            Some((_, n)) => *n = n.wrapping_add(by),
            None => {
                if o.len() == OWN_RUNTIMES {
                    o.remove(0);
                }
                o.push((rt, by));
            }
        }
    });
}

const OWN_RUNTIMES: usize = 8;

/// The episodes this thread has landed on `rt`.
pub(crate) fn own_episodes(rt: RuntimeId) -> u64 {
    OWN_EPISODES.with(|o| {
        o.borrow()
            .iter()
            .find(|(r, _)| *r == rt)
            .map_or(0, |&(_, n)| n)
    })
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

/// This thread's wound-wait ticket, 0 until its first registration.
pub(crate) fn ticket_now() -> u64 {
    ENTRY.with(|e| e.ticket.get())
}

/// Whether a take-over of this thread's outermost entry met a claim.
#[inline(always)]
pub(crate) fn any_taken() -> bool {
    ENTRY.with(|e| e.taken.get())
}

pub(crate) fn set_any_taken(taken: bool) {
    ENTRY.with(|e| e.taken.set(taken));
}

pub(crate) fn set_wounded(wounded: bool) {
    ENTRY.with(|e| e.wounded.set(wounded));
}

pub(crate) fn wounded() -> bool {
    ENTRY.with(|e| e.wounded.get())
}

/// Checks, at an outermost entry's exit, that the thread carries no wound
/// and holds no claim into its next entry.
pub(crate) fn debug_assert_exit_clear() {
    debug_assert!(
        !wounded(),
        "incr-core: an outermost entry exits with its thread still wounded"
    );
    debug_assert!(
        !held_any() || held_keys().is_empty(),
        "incr-core: an outermost entry exits holding claims"
    );
}

/// Marks a `Shared` runtime's key; runtime ids stay far below 2^63.
const SHARED_KEY: u64 = 1 << 63;

/// One count on the thread's engine entry depth; dropping it, by an
/// unwind included, gives the count back, and the outermost entry's drop
/// gives up the thread's ticket. An entry of another runtime than the
/// innermost one's keeps the key of the entry below it in [`OUTER_RTS`],
/// so that the runtimes open below are known.
pub(crate) struct EntryDepth {
    outermost: bool,
    switched: bool,
    first_for_runtime: bool,
    shared_outermost: bool,
}

/// The keys below this thread's runtime switches, innermost last, with a
/// count per key and of the `Shared` ones, so that a switch costs the
/// same however many are open.
#[derive(Default)]
struct OuterRts {
    stack: Vec<u64>,
    open: HashMap<u64, u32, FxBuildHasher>,
    shared: u32,
}

impl EntryDepth {
    /// An entry of runtime `rt`, a `Shared` one if `shared`.
    #[inline(always)]
    pub(crate) fn enter(rt: RuntimeId, shared: bool) -> Self {
        let base = 0_u8;
        let base = std::ptr::from_ref(&base) as usize;
        let key = rt.get() | if shared { SHARED_KEY } else { 0 };
        ENTRY.with(|e| {
            let depth = e.depth.get();
            e.depth.set(depth + 1);
            if depth == 0 {
                e.cur_rt.set(key);
                e.stack_base.set(base);
                e.budget_clock
                    .set(BUDGET_CLOCK.load(std::sync::atomic::Ordering::Relaxed));
                #[cfg(all(test, not(loom)))]
                OUTERMOST_ENTRIES.with(|n| n.set(n.get() + 1));
                Self {
                    outermost: true,
                    switched: false,
                    first_for_runtime: true,
                    shared_outermost: shared,
                }
            } else if e.cur_rt.get() == key {
                Self {
                    outermost: false,
                    switched: false,
                    first_for_runtime: false,
                    shared_outermost: false,
                }
            } else {
                switch_in(e, key)
            }
        })
    }

    #[inline(always)]
    pub(crate) fn is_outermost(&self) -> bool {
        self.outermost
    }

    /// No entry of this runtime is open below this one on the thread.
    #[inline(always)]
    #[cfg_attr(any(not(test), loom), allow(dead_code))]
    pub(crate) fn first_for_runtime(&self) -> bool {
        self.first_for_runtime
    }

    /// The entry is a `Shared` runtime's, and no `Shared` runtime's entry
    /// is open below it on the thread.
    #[inline(always)]
    pub(crate) fn shared_outermost(&self) -> bool {
        self.shared_outermost
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
                if e.taken.get() {
                    crate::reserve::clear_taken();
                }
            }
            if self.switched {
                switch_out(e);
            }
        });
        // Each runtime's first entry on the thread ends its holds, and
        // the outermost is every runtime's first.
        debug_assert!(
            !self.outermost || std::thread::panicking() || !crate::hold::active(),
            "incr-core: a held landing outlives the thread's outermost entry"
        );
    }
}

#[cold]
#[inline(never)]
fn switch_in(e: &EntryState, key: u64) -> EntryDepth {
    let below = e.cur_rt.get();
    let pushed = OUTER_RTS.try_with(|o| {
        let mut o = o.borrow_mut();
        o.stack.push(below);
        *o.open.entry(below).or_insert(0) += 1;
        o.shared += u32::from(below & SHARED_KEY != 0);
        (!o.open.contains_key(&key), o.shared == 0)
    });
    match pushed {
        Ok((first_for_runtime, no_shared_below)) => {
            e.cur_rt.set(key);
            EntryDepth {
                outermost: false,
                switched: true,
                first_for_runtime,
                shared_outermost: key & SHARED_KEY != 0 && no_shared_below,
            }
        }
        // An entry from a destructor of this thread's locals, after
        // `OUTER_RTS`'s: it is taken as a nested entry of the runtime below.
        Err(_) => EntryDepth {
            outermost: false,
            switched: false,
            first_for_runtime: false,
            shared_outermost: false,
        },
    }
}

#[cold]
#[inline(never)]
fn switch_out(e: &EntryState) {
    let below = OUTER_RTS.with(|o| {
        let mut o = o.borrow_mut();
        let below = o
            .stack
            .pop()
            .expect("incr-core: a switched entry keeps the key below it");
        match o.open.get_mut(&below) {
            Some(n) if *n > 1 => *n -= 1,
            _ => {
                o.open.remove(&below);
            }
        }
        o.shared -= u32::from(below & SHARED_KEY != 0);
        below
    });
    e.cur_rt.set(below);
}

/// Whether an engine entry of `rt` is open on this thread.
pub(crate) fn open_entry_of(rt: RuntimeId) -> bool {
    let rt = rt.get();
    ENTRY.with(|e| e.depth.get() != 0 && (e.cur_rt.get() & !SHARED_KEY == rt || outer_open(rt)))
}

/// Whether an entry of `rt` made now is the first inside another entry on
/// the thread: [`open_entry_of`] is false under a nonzero depth.
pub(crate) fn first_nested_of(rt: RuntimeId) -> bool {
    let rt = rt.get();
    ENTRY.with(|e| e.depth.get() != 0 && e.cur_rt.get() & !SHARED_KEY != rt && !outer_open(rt))
}

/// Notes `rec` as the group record this thread left at `slot`.
pub(crate) fn note_rec_left(rt: RuntimeId, slot: u32, rec: Option<Arc<GroupInfo>>) {
    OUTER.with(|o| o.borrow_mut().recs_left.push((rt, slot, rec)));
    ENTRY.with(|e| e.outer_set.set(true));
}

/// How many group record changes this thread has noted in this outermost
/// entry.
pub(crate) fn recs_left_count() -> usize {
    OUTER.with(|o| o.borrow().recs_left.len())
}

/// Whether `current` is the group record this thread last left at `slot`,
/// among the changes it noted after its first `since`.
pub(crate) fn left_rec_since(
    since: usize,
    rt: RuntimeId,
    slot: u32,
    current: Option<&Arc<GroupInfo>>,
) -> bool {
    OUTER.with(|o| {
        o.borrow()
            .recs_left
            .get(since..)
            .unwrap_or_default()
            .iter()
            .rev()
            .find(|(r, s, _)| *r == rt && *s == slot)
            .is_some_and(|(_, _, left)| match (left, current) {
                (Some(l), Some(c)) => Arc::ptr_eq(l, c),
                (None, None) => true,
                _ => false,
            })
    })
}

/// Whether `rt` has an open entry on this thread as a `Shared` runtime.
pub(crate) fn shared_open(rt: RuntimeId) -> bool {
    let key = rt.get() | SHARED_KEY;
    ENTRY.with(|e| e.depth.get() != 0 && e.cur_rt.get() == key)
        || OUTER_RTS.with(|o| o.borrow().open.contains_key(&key))
}

fn outer_open(rt: u64) -> bool {
    OUTER_RTS.with(|o| {
        let o = o.borrow();
        !o.open.is_empty() && (o.open.contains_key(&rt) || o.open.contains_key(&(rt | SHARED_KEY)))
    })
}

/// Resets the state an outermost entry starts from.
#[inline(always)]
pub(crate) fn reset_outermost() {
    if ENTRY.with(|e| e.outer_set.replace(false)) {
        ENTRY.with(|e| e.cycle_caught.set(false));
        OUTER.with(|o| {
            let mut o = o.borrow_mut();
            o.dissolved.clear();
            o.dissolved_groups.clear();
            o.group_blockers.clear();
            o.recs_left.clear();
            o.last_cycle = None;
            o.budgets_met.clear();
            o.failed_here.clear();
            ENTRY.with(|e| e.failed_here.set(0));
        });
    }
}

#[cfg(all(test, not(loom)))]
#[path = "cycle_state_tests.rs"]
mod tests;
