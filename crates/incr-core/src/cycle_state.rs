//! Per-thread cycle state shared by every runtime on the thread: the
//! claims this thread holds outside its frame stack (the held map), the
//! engine entry depth, the kind of an engine unwind in flight, the last
//! cycle an uncaught cycle panic reported, and the roots whose episodes
//! dissolved during the current outermost entry.

use std::borrow::{Borrow, BorrowMut};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::ops::{Deref, DerefMut};

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
    pub(crate) s: Option<Erased>,
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
    poisoned: bool,
    /// A run of this claim panicked in user code.
    pub(crate) failed: bool,
    /// `Cycle(M)` while the node is a member of a running episode.
    pub(crate) episode: Option<Cycle>,
    /// That episode's id, 0 when there is none.
    pub(crate) ep: u64,
    /// The member is in the episode's set of unseeded members that read
    /// each other.
    pub(crate) in_cu: bool,
    /// While the buffer and run record come from a run made under
    /// speculation: the record from before that run's write.
    pre_spec: Option<Box<RunRecord>>,
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
            poisoned: false,
            failed: false,
            episode: None,
            ep: 0,
            in_cu: false,
            pre_spec: None,
        }
    }

    /// Records a finished run: its deps, buffer, whether it reached the
    /// child index and whether it was poisoned. Under speculation it first
    /// keeps the record it replaces; outside, it settles first. Returns
    /// the buffer it replaces and the value a settle dropped, for the
    /// caller to drop outside the held map.
    pub(crate) fn record(
        &mut self,
        look: Look,
        deps: Vec<(NodeId, u32)>,
        buf: Option<Erased>,
        opened: bool,
        poisoned: bool,
    ) -> (Option<Erased>, Option<Unbuilt>) {
        let spec_buf = if look.speculative {
            self.keep_record();
            None
        } else {
            self.settle()
        };
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
        self.poisoned |= poisoned;
        (std::mem::replace(&mut self.buf, buf), spec_buf)
    }

    /// The claim's run as `look` sees it, and the value a settle dropped,
    /// for the caller to drop outside the held map.
    pub(crate) fn look(&mut self, look: Look) -> (Seen<&mut Held>, Option<Unbuilt>) {
        let spec_buf = self.settle_for(look);
        (Seen(self), spec_buf)
    }

    /// [`Self::look`] at a claim out of the held map.
    pub(crate) fn into_look(mut self, look: Look) -> (Seen<Held>, Option<Unbuilt>) {
        let spec_buf = self.settle_for(look);
        (Seen(self), spec_buf)
    }

    /// Takes the buffer to drop it, leaving the run record as it is.
    pub(crate) fn discard_buf(&mut self) -> Option<Discarded> {
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
        self.pre_spec.is_some()
    }

    /// Keeps a speculative run's buffer and record as the claim's own, for
    /// a landing that speculation may make.
    pub(crate) fn adopt(&mut self) {
        self.pre_spec = None;
    }

    /// Before a run's buffer and record are written under speculation:
    /// keeps the record they replace, unless an earlier speculative run's
    /// write already did.
    fn keep_record(&mut self) {
        if self.pre_spec.is_none() {
            self.pre_spec = Some(Box::new(RunRecord {
                deps: self.deps.clone(),
                last: self.last.clone(),
                installed: self.installed.clone(),
                opened: self.opened,
                poisoned: self.poisoned,
            }));
        }
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
    fn settle(&mut self) -> Option<Unbuilt> {
        let r = self.pre_spec.take()?;
        self.deps = r.deps;
        self.last = r.last;
        self.installed = r.installed;
        self.opened = r.opened;
        self.poisoned = r.poisoned;
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
    #[cfg(all(test, not(loom)))]
    const SPECULATIVE: Self = Self { speculative: true };
    #[cfg(all(test, not(loom)))]
    const REAL: Self = Self { speculative: false };

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

    pub(crate) fn poisoned(&self) -> bool {
        self.0.borrow().poisoned
    }
}

impl<H: BorrowMut<Held>> Seen<H> {
    pub(crate) fn take_buf(&mut self) -> Option<Erased> {
        self.0.borrow_mut().buf.take()
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

/// A claim's run record, kept while a speculative run's is written over it.
struct RunRecord {
    deps: Vec<(NodeId, u32)>,
    last: Vec<(NodeId, u32)>,
    installed: Vec<NodeId>,
    opened: bool,
    poisoned: bool,
}

/// An episode between member runs. A doom that discards a member run
/// suspends it in a replay context until that run's virtual frame pops.
pub(crate) struct Episode {
    pub(crate) id: u64,
    pub(crate) root: NodeId,
    pub(crate) cidx: u32,
    pub(crate) validated: bool,
    /// The landing's one cut to the root's SCC in the final reads has run.
    pub(crate) pruned: bool,
    pub(crate) members: Vec<u32>,
    pub(crate) cu: Vec<u32>,
    /// The iteration state since the last restart, `None` at a restart.
    pub(crate) round: Option<Round>,
    /// The root's compute is a walk's entry, which contains the episode's
    /// abort.
    pub(crate) walked: bool,
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
        Some(e)
    }

    pub(crate) fn get(&self, key: &(RuntimeId, u32)) -> Option<&Held> {
        self.claims.get(key)
    }

    pub(crate) fn get_mut(&mut self, key: &(RuntimeId, u32)) -> Option<&mut Held> {
        self.claims.get_mut(key)
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
    /// `BUDGET_CLOCK` when the outermost entry began: a runtime whose
    /// budget changed later may have frames on the stack that read under
    /// the budget before.
    budget_clock: Cell<u64>,
}

/// What the outermost entry resets: the roots whose episodes dissolved,
/// and the cycle the last uncaught cycle panic reported.
struct Outer {
    dissolved: Vec<(RuntimeId, u32)>,
    /// The landed group of each root whose episode dissolved in this
    /// entry, held so that no later record can reuse its address.
    dissolved_groups: Vec<(RuntimeId, u32, Arc<GroupInfo>)>,
    last_cycle: Option<Cycle>,
    /// Each runtime whose budget change this thread has met, with the
    /// `BUDGET_CLOCK` count of the change.
    budgets_met: Vec<(RuntimeId, u64)>,
}

#[cfg(not(loom))]
thread_local! {
    static HELD: RefCell<HeldMap> = RefCell::new(HeldMap::new());
    static ENTRY: EntryState = const {
        EntryState {
            depth: Cell::new(0),
            ticket: Cell::new(0),
            wounded: Cell::new(false),
            failures: Cell::new(0),
            outer_set: Cell::new(false),
            held_any: Cell::new(false),
            stack_base: Cell::new(0),
            budget_clock: Cell::new(0),
        }
    };
    static OWN_EPISODES: RefCell<Vec<(RuntimeId, u64)>> = const { RefCell::new(Vec::new()) };
    static OUTER: RefCell<Outer> = const {
        RefCell::new(Outer {
            dissolved: Vec::new(),
            dissolved_groups: Vec::new(),
            last_cycle: None,
            budgets_met: Vec::new(),
        })
    };
}

#[cfg(loom)]
loom::thread_local! {
    static HELD: RefCell<HeldMap> = RefCell::new(HeldMap::new());
    static ENTRY: EntryState = EntryState {
        depth: Cell::new(0),
        ticket: Cell::new(0),
        wounded: Cell::new(false),
        failures: Cell::new(0),
        outer_set: Cell::new(false),
        held_any: Cell::new(false),
        stack_base: Cell::new(0),
        budget_clock: Cell::new(0),
    };
    static OWN_EPISODES: RefCell<Vec<(RuntimeId, u64)>> = RefCell::new(Vec::new());
    static OUTER: RefCell<Outer> = RefCell::new(Outer {
        dissolved: Vec::new(),
        dissolved_groups: Vec::new(),
        last_cycle: None,
        budgets_met: Vec::new(),
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

/// For a binding that runs user code in another language: whether an
/// engine unwind is in flight on this thread, or a wound has marked it for
/// a retry. An error that code raises now belongs to a run the engine
/// discards, so the binding must not keep it to report later.
#[doc(hidden)]
pub fn engine_unwind_in_progress() -> bool {
    unwind_kind().is_some() || wounded()
}

/// A cycle unwind a binding carries through user code in another
/// language, set aside while that code makes an engine call, such as a
/// read in a Python `finally` the unwind runs.
#[doc(hidden)]
#[must_use]
pub struct CycleUnwindAside(());

/// For a binding whose user code runs while it carries a cycle unwind:
/// clears the unwind kind so an engine call that code makes computes,
/// fails and finds its own cycles as if no unwind were in flight. Without
/// this, an engine catch inside the call takes the carried cycle as the
/// call's own. `None` unless a cycle unwind is in flight; a wound or a
/// discard is never set aside, since the run it belongs to is given up.
#[doc(hidden)]
pub fn set_aside_cycle_unwind() -> Option<CycleUnwindAside> {
    if wounded() || unwind_kind() != Some(UnwindKind::Cycle) {
        return None;
    }
    set_unwind_kind(None);
    Some(CycleUnwindAside(()))
}

impl CycleUnwindAside {
    /// Ends the engine call: `true` if the carried cycle unwind is in
    /// flight again, `false` if the call ended in an engine unwind of its
    /// own, which replaces it as the exception it raises replaces the
    /// carried one.
    pub fn put_back(self) -> bool {
        if engine_unwind_in_progress() {
            return false;
        }
        set_unwind_kind(Some(UnwindKind::Cycle));
        true
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

/// The stack address the outermost engine entry recorded.
pub(crate) fn stack_base() -> usize {
    ENTRY.with(|e| e.stack_base.get())
}

/// The thread's engine entry depth: 0 outside every engine entry.
pub(crate) fn depth() -> u32 {
    ENTRY.with(|e| e.depth.get())
}

/// Counts an episode this thread lands on `rt`, for the last
/// `OWN_RUNTIMES` runtimes it landed one on. A count dropped or restarted
/// only makes a walk's own share differ from all landings, which re-plans.
pub(crate) fn own_episode(rt: RuntimeId) {
    OWN_EPISODES.with(|o| {
        let mut o = o.borrow_mut();
        match o.iter_mut().find(|(r, _)| *r == rt) {
            Some((_, n)) => *n = n.wrapping_add(1),
            None => {
                if o.len() == OWN_RUNTIMES {
                    o.remove(0);
                }
                o.push((rt, 1));
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
                e.budget_clock
                    .set(BUDGET_CLOCK.load(std::sync::atomic::Ordering::Relaxed));
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
            o.dissolved_groups.clear();
            o.last_cycle = None;
            o.budgets_met.clear();
        });
    }
}

#[cfg(all(test, not(loom)))]
mod binding_tests {
    use super::*;

    #[test]
    fn engine_unwind_in_progress_covers_every_kind_and_a_wound() {
        assert!(!engine_unwind_in_progress());
        for kind in [UnwindKind::Cycle, UnwindKind::Wound, UnwindKind::Discard] {
            set_unwind_kind(Some(kind));
            assert!(engine_unwind_in_progress());
            set_unwind_kind(None);
        }
        set_wounded(true);
        assert!(engine_unwind_in_progress());
        set_wounded(false);
        assert!(!engine_unwind_in_progress());
    }

    #[test]
    fn only_a_cycle_unwind_is_set_aside() {
        assert!(set_aside_cycle_unwind().is_none());
        for kind in [UnwindKind::Wound, UnwindKind::Discard] {
            set_unwind_kind(Some(kind));
            assert!(set_aside_cycle_unwind().is_none());
            assert_eq!(unwind_kind(), Some(kind));
            set_unwind_kind(None);
        }
        set_unwind_kind(Some(UnwindKind::Cycle));
        set_wounded(true);
        assert!(set_aside_cycle_unwind().is_none());
        set_wounded(false);

        let aside = set_aside_cycle_unwind().expect("a cycle unwind is set aside");
        assert!(!engine_unwind_in_progress());
        assert!(aside.put_back());
        assert_eq!(unwind_kind(), Some(UnwindKind::Cycle));

        let aside = set_aside_cycle_unwind().expect("a cycle unwind is set aside");
        set_unwind_kind(Some(UnwindKind::Discard));
        assert!(!aside.put_back());
        assert_eq!(unwind_kind(), Some(UnwindKind::Discard));
        set_unwind_kind(None);
    }
}

#[cfg(all(test, not(loom)))]
#[path = "cycle_state_tests.rs"]
mod tests;
