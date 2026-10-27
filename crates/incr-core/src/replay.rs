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

use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::hash_map::Entry as MapEntry;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
use std::rc::Rc;

use crate::changes::Count;
use crate::cycle_state::{held_episode, Episode, UnwindKind};
use crate::deferred::Unbuilt;
use crate::few::Few;
use crate::fx::FxBuildHasher;
use crate::handle::RuntimeId;
use crate::notes::{Cut, CutLandings, Landed, Log, Marked};
use crate::state::NodeState;
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
    /// It stands for a walk's entry, so its compute contains a panic as
    /// the walk's does.
    pub(crate) walked: bool,
    /// An R5 read of a member this thread holds doomed it, so its run is
    /// that read's member run, not a stand-in for a frame's compute.
    pub(crate) r5: bool,
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
    /// The frames below and the reads they were in when this frame ran, as
    /// a hash of their [`Step`]s: a frame of the same node at the same
    /// position under another frame or read below is another run.
    pub(crate) path: u64,
    /// The seq of the frame's run (see [`Context::runs`]).
    pub(crate) seq: u32,
    /// The level of the entry whose run made it.
    pub(crate) maker: u32,
    /// For an episode's member run, the episode, which waits in the
    /// context until this frame pops.
    pub(crate) ep: Option<u64>,
    /// For a member run, the episode its member is in, R5's nested runs
    /// included: the frame's [`ReadKey::episode`].
    pub(crate) member_of: Option<u64>,
    /// The compute the frame below was blocked reading, `slot |
    /// generation << 32`: the frame's own node, or for an episode's member
    /// run the episode's root, whose compute ran the episode.
    pub(crate) computes: u64,
    /// The claims its run deferred before the doom, as `(slot, anchor)`.
    /// The discard released them, but in the unbounded run they are still
    /// held, so a cycle closed above counts them among its participants.
    pub(crate) deferred: Vec<(u32, u32)>,
    /// A walk computed the frame's node, so the walk contains an unwind
    /// out of its run.
    pub(crate) walked: bool,
}

/// The read a delivery is for.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ReadKey {
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
    /// The frame's run's seq (see [`Context::runs`]): another run of the
    /// frame at the position along the path reads none of this one's
    /// deliveries.
    pub(crate) seq: u32,
    /// For a member run, the episode its member is in: a discovery run of
    /// the node at the position, and the runs of another episode, are other
    /// runs.
    pub(crate) episode: Option<u64>,
    /// The runtime of the node `reading` names, when it is not the
    /// frame's: a read of another runtime's node.
    pub(crate) foreign: Option<RuntimeId>,
}

impl ReadKey {
    /// It is for the read `last`, as `(reading, ordinal)`, of the frame of
    /// `node` at `cidx`, on any path.
    fn read_by(&self, node: u32, generation: u32, cidx: u32, last: (u64, u32)) -> bool {
        self.node == node
            && self.generation == generation
            && self.cidx == cidx
            && (self.reading, self.ordinal) == last
    }

    /// The read virtual frame `v` was blocked in.
    pub(crate) fn blocked(v: &VFrame) -> Self {
        Self {
            node: v.node,
            generation: v.generation,
            cidx: v.cidx,
            reading: v.reading,
            ordinal: v.ordinal,
            path: v.path,
            seq: v.seq,
            episode: v.member_of,
            foreign: None,
        }
    }
}

/// What an entry's run left for the frame below the entry's level, whose
/// retry gets it at its read of the entry's node, as the unbounded run's
/// frame gets it at that read.
pub(crate) struct Delivery {
    pub(crate) key: ReadKey,
    /// The level of the entry whose retry recomputes the frame; the
    /// delivery is dropped when that run ends without reaching the read.
    pub(crate) maker: u32,
    /// The frame's current retry has read it: a later read of the node in
    /// the same retry is an ordinary one, as in the unbounded run. Private
    /// so that [`Pending::set_taken`], which keeps the untaken counts, is
    /// its only writer.
    taken: bool,
    /// The outcome of the read the frame was blocked in, which a later
    /// outcome of the same read replaces; otherwise an earlier read's.
    pub(crate) blocked: bool,
    pub(crate) what: Delivered,
    /// For a user panic whose node landed Failed above the frame, the
    /// record that tells which landings of the frame's node by other
    /// threads let that failure out.
    pub(crate) provenance: Option<Rc<dyn Provenance>>,
    /// With a provenance record, the virtual frames below the frame's that
    /// have popped since, contiguous and highest first, as `(cidx, node,
    /// generation)`: a read of one of their nodes stands for its frame's
    /// run.
    pub(crate) below: Vec<(u32, u32, u32)>,
    /// The copy of the payload the latest take of it raised.
    pub(crate) resumed: Option<PayloadKey>,
    /// The landing, Dirty or Failed, that the payload's first take out of
    /// the frame's compute gave the frame's node. A later take, in a retry
    /// doomed again, lands it so again: the deps it finds may have landed
    /// since in ways the unbounded run's frame never saw.
    pub(crate) landed: Option<NodeState>,
    /// A run of the frame has read past it: the landings it carries stood
    /// as notes in the unbounded run from then on, so a later pop above
    /// the frame retires them as it retires notes.
    pub(crate) passed: bool,
    /// For a `Walk` or `Direct`, the runs ended above the frame in its read
    /// up to where the retry resumes it, which that read starts from.
    pub(crate) runs: RunSlot,
}

/// The runs at one position along `path` that have ended undoomed in the
/// current read of the frame below, as `(node, generation, count)`. Slots
/// are found by position and path alone (see [`Context::runs`]). `owner`
/// is the reading frame's depth on the thread's stack, which only decides
/// how long [`start_read`] keeps the slot; it is 0 for a slot no read set,
/// which a run's end makes under a virtual frame's blocked read or the
/// opener's.
#[derive(Clone, Default)]
pub(crate) struct RunSlot {
    pub(crate) path: u64,
    pub(crate) owner: u32,
    pub(crate) counts: Few<(u32, u32, u32)>,
}

impl RunSlot {
    fn count(&self, node: u32, generation: u32) -> u32 {
        (self.counts.as_slice().iter())
            .find(|&&(n, g, _)| (n, g) == (node, generation))
            .map_or(0, |&(_, _, k)| k)
    }

    fn bump(&mut self, node: u32, generation: u32) {
        match (self.counts.as_mut_slice().iter_mut())
            .find(|(n, g, _)| (*n, *g) == (node, generation))
        {
            Some((_, _, k)) => *k += 1,
            None => self.counts.push((node, generation, 1)),
        }
    }
}

/// A context's run slots by position, the first at its `boundary + 1`; a
/// position holds a slot per path its reads were made along.
#[derive(Default)]
pub(crate) struct RunSlots(Vec<Few<RunSlot>>);

impl RunSlots {
    fn at(&self, i: usize) -> &[RunSlot] {
        self.0.get(i).map_or(&[], Few::as_slice)
    }

    fn at_mut(&mut self, i: usize) -> &mut Few<RunSlot> {
        if self.0.len() <= i {
            self.0.resize_with(i + 1, Few::default);
        }
        &mut self.0[i]
    }

    fn get(&self, i: usize, path: u64) -> Option<&RunSlot> {
        self.at(i).iter().find(|s| s.path == path)
    }

    fn entry(&mut self, i: usize, path: u64) -> &mut RunSlot {
        let slots = self.at_mut(i);
        let j = match slots.as_slice().iter().position(|s| s.path == path) {
            Some(j) => j,
            None => {
                slots.push(RunSlot {
                    path,
                    owner: 0,
                    counts: Few::None,
                });
                slots.as_slice().len() - 1
            }
        };
        &mut slots.as_mut_slice()[j]
    }

    /// The slots from index `i` on, which this keeps no longer.
    fn split_off(&mut self, i: usize) -> Self {
        Self(if i < self.0.len() {
            self.0.split_off(i)
        } else {
            Vec::new()
        })
    }
}

/// The Failed landings noted for a payload that a copy of it carries, each
/// with its depth above the reading frame, and the payload they were noted
/// for.
#[derive(Clone, Default)]
pub(crate) struct Landings {
    pub(crate) source: Option<PayloadKey>,
    nodes: Nodes,
}

#[derive(Clone)]
enum Nodes {
    Owned(Vec<Landed>),
    /// The notes of `log` the copy was taken of: its first `len` entries
    /// but those gone before removal `epoch` and those `cut` took, at
    /// their depths above `base`, none deeper than `depth`.
    Shared {
        log: Rc<RefCell<Log>>,
        len: usize,
        epoch: u64,
        base: u32,
        depth: u32,
        cut: Option<Cut>,
    },
}

impl Default for Nodes {
    fn default() -> Self {
        Nodes::Owned(Vec::new())
    }
}

/// The fields of a [`Nodes::Shared`] copy.
pub(crate) struct SharedCopy<'a> {
    pub(crate) log: &'a Rc<RefCell<Log>>,
    pub(crate) len: usize,
    pub(crate) epoch: u64,
    pub(crate) base: u32,
    pub(crate) cut: Option<&'a Cut>,
}

impl Landings {
    pub(crate) fn owned(source: Option<PayloadKey>, nodes: Vec<Landed>) -> Self {
        Self {
            source,
            nodes: Nodes::Owned(nodes),
        }
    }

    /// A copy of the notes in the first `len` entries of `log` still
    /// noted at removal `epoch`, none above `top`.
    pub(crate) fn shared(
        source: Option<PayloadKey>,
        log: Rc<RefCell<Log>>,
        epoch: u64,
        base: u32,
        top: u32,
    ) -> Self {
        let len = log.borrow().entries.len();
        Self {
            source,
            nodes: Nodes::Shared {
                log,
                len,
                epoch,
                base,
                depth: top.saturating_sub(base),
                cut: None,
            },
        }
    }

    fn owned_nodes(&mut self) -> &mut Vec<Landed> {
        if let Nodes::Shared {
            log,
            len,
            epoch,
            base,
            cut,
            ..
        } = &self.nodes
        {
            let nodes = log.borrow().landings(*len, *epoch, *base, cut.as_ref());
            self.nodes = Nodes::Owned(nodes);
        }
        match &mut self.nodes {
            Nodes::Owned(nodes) => nodes,
            Nodes::Shared { .. } => unreachable!(),
        }
    }

    fn holds(&mut self, node: (u32, u32)) -> bool {
        let nodes = self.owned_nodes();
        note_work(nodes.len());
        nodes.iter().any(|&(n, _, _)| n == node)
    }

    /// A copy that still shares its log, as [`Nodes::Shared`] holds it.
    pub(crate) fn as_shared(&self) -> Option<SharedCopy<'_>> {
        match &self.nodes {
            Nodes::Shared {
                log,
                len,
                epoch,
                base,
                cut,
                ..
            } => Some(SharedCopy {
                log,
                len: *len,
                epoch: *epoch,
                base: *base,
                cut: cut.as_ref(),
            }),
            Nodes::Owned(_) => None,
        }
    }

    pub(crate) fn into_nodes(mut self) -> Vec<Landed> {
        std::mem::take(self.owned_nodes())
    }

    /// A bound on the depth of the deepest landing the copy still holds:
    /// exact for an owned copy, `None` once it holds none; a shared copy
    /// answers its depth unread, even once its notes are all removed.
    fn deepest(&self) -> Option<u32> {
        match &self.nodes {
            Nodes::Owned(nodes) => {
                note_work(nodes.len());
                nodes.iter().map(|&(_, depth, _)| depth).max()
            }
            // No landing is deeper than `depth`, and finding the deepest
            // would take each delivery of a deep chain's copy apart.
            Nodes::Shared { depth, .. } => Some(*depth),
        }
    }

    /// A bound on the deepest landing's depth, without reading a shared
    /// copy's log.
    fn depth(&self) -> u32 {
        match &self.nodes {
            Nodes::Owned(nodes) => {
                note_work(nodes.len());
                nodes.iter().map(|&(_, depth, _)| depth).max().unwrap_or(0)
            }
            Nodes::Shared { depth, .. } => *depth,
        }
    }

    /// Applies `ev` to the landings of a read by the frame at `cidx`, as it
    /// applies to the notes they stand for; `resumed` is the copy the
    /// frame's latest take raised, which counts as the source.
    fn retire(&mut self, cidx: u32, ev: &Retire, resumed: Option<PayloadKey>) {
        let source = self.source;
        let ours = |p: PayloadKey| source == Some(p) || resumed == Some(p);
        let gone: Box<dyn Fn((u32, u32), u32) -> bool> = match *ev {
            Retire::Above(at) => Box::new(move |_, pos| pos > at),
            Retire::Contained { cidx: at, payload } => {
                let same = ours(payload);
                Box::new(move |_, pos| pos > at || pos == at && same)
            }
            Retire::Landing {
                cidx: at,
                payload,
                node,
            } => {
                let other = payload.is_some_and(|p| !ours(p));
                Box::new(move |n, pos| n == node || pos > at && other)
            }
        };
        let nodes = self.owned_nodes();
        note_work(nodes.len());
        nodes.retain(|&(n, depth, _)| !gone(n, cidx + 1 + depth));
    }
}

/// An event that retires notes (`runtime_contain`), applied to the copies
/// of them that kept reads and passed deliveries hold.
#[derive(Clone, Copy)]
pub(crate) enum Retire {
    /// A frame at this position popped with a value.
    Above(u32),
    /// A walk's entry at `cidx` contained `payload`.
    Contained { cidx: u32, payload: PayloadKey },
    /// `node` landed Failed at `cidx`, unwinding `payload` if known.
    Landing {
        cidx: u32,
        payload: Option<PayloadKey>,
        node: (u32, u32),
    },
}

impl Retire {
    fn cidx(&self) -> u32 {
        match *self {
            Retire::Above(at)
            | Retire::Contained { cidx: at, .. }
            | Retire::Landing { cidx: at, .. } => at,
        }
    }

    fn node(&self) -> Option<(u32, u32)> {
        match *self {
            Retire::Landing { node, .. } => Some(node),
            _ => None,
        }
    }
}

/// A context's deliveries. Each keeps its index until it is removed, and
/// is found by its read, the level whose run made it, its frame, its
/// frame's node, or the provenance record it carries, without a pass over
/// the others: a deep chain's context holds a delivery per level, and its
/// reads look them up per level.
#[derive(Default)]
pub(crate) struct Pending {
    slots: Vec<Option<Delivery>>,
    live: usize,
    by_key: Index<ReadKey, usize>,
    /// A removal leaves its index in its maker's list until the list holds
    /// as many removed as live, so a removal from a budget-long list
    /// passes over none of it; readers skip emptied slots, which no push
    /// reuses while a list holds them.
    by_maker: HashMap<u32, Made, FxBuildHasher>,
    by_frame: Index<(u32, u32, u32), usize>,
    /// By the frame and the node it reads.
    by_read: Index<(u32, u32, u32, u64), usize>,
    /// How many deliveries each frame's own runtime's reads have still to
    /// take: those with no `foreign` key and not `taken`, kept by `push`,
    /// `unindex` and `set_taken`.
    untaken: HashMap<(u32, u32, u32), u32, FxBuildHasher>,
    /// By the frame's node and generation, at any position.
    by_node: Index<(u32, u32), usize>,
    /// How many deliveries carry a provenance record.
    traced: usize,
    /// The deliveries with a record, by each frame a read of whose node
    /// they stand for, as `(cidx, node, generation)`: their own and those
    /// in their `below`.
    by_stand: Index<(u32, u32, u32), usize>,
    /// The deliveries with a record, by the position of the lowest frame
    /// they stand for, which the pop of the virtual frame below extends.
    by_frontier: Index<u32, usize>,
    /// The passed deliveries with a landing deeper than just above their
    /// frame, by the highest position one of their landings may have.
    by_top: BTreeSet<(u32, usize)>,
    /// The passed deliveries whose landings are all just above their frame,
    /// by that position and each payload they stand for: short of a copied
    /// node's landing, only a walk's entry there containing that payload
    /// retires them.
    by_payload: BTreeSet<(u32, PayloadKey, usize)>,
    filed: HashMap<usize, Filed, FxBuildHasher>,
}

/// Where a passed delivery is filed for the events that retire its
/// landings (`Pending::retire`).
#[derive(Clone, Copy)]
enum Filed {
    Deep(u32),
    Shallow(u32, [Option<PayloadKey>; 2]),
}

/// The deliveries a level's run made, oldest first, and how many of them
/// are live.
#[derive(Default)]
struct Made {
    live: u32,
    list: Few<usize>,
}

impl Made {
    /// One of its entries went: whether none is left, else the list sheds
    /// the indexes `keep` refuses once it holds twice as many as live, so a
    /// removal from a long list passes over none of it.
    fn shed(&mut self, keep: impl FnMut(&usize) -> bool) -> bool {
        self.live -= 1;
        if self.live == 0 {
            return true;
        }
        if self.list.as_slice().len() >= 2 * self.live as usize {
            note_work(self.list.as_slice().len());
            self.list.retain(keep);
        }
        false
    }
}

fn frame_of(key: &ReadKey) -> (u32, u32, u32) {
    (key.node, key.generation, key.cidx)
}

/// The position of the lowest frame `d` stands for a read of.
fn frontier(d: &Delivery) -> u32 {
    d.below.last().map_or(d.key.cidx, |b| b.0)
}

fn read_of(key: &ReadKey) -> (u32, u32, u32, u64) {
    (key.node, key.generation, key.cidx, key.reading)
}

fn awaits(d: &Delivery) -> bool {
    !d.taken && d.key.foreign.is_none()
}

type Index<K, T> = HashMap<K, Few<T>, FxBuildHasher>;

fn list<K: std::hash::Hash + Eq, T>(map: &mut Index<K, T>, at: K, x: T) {
    match map.entry(at) {
        MapEntry::Occupied(mut e) => e.get_mut().push(x),
        MapEntry::Vacant(e) => {
            e.insert(Few::One(x));
        }
    }
}

/// Drops the items `gone` accepts from the list `map` holds under `at`.
fn unlist<K: std::hash::Hash + Eq, T>(map: &mut Index<K, T>, at: &K, gone: impl Fn(&T) -> bool) {
    if map.get_mut(at).is_some_and(|l| {
        note_work(l.as_slice().len());
        l.retain(|x| !gone(x))
    }) {
        map.remove(at);
    }
}

/// Drops the items `gone` accepts from each list `map` holds under `keys`,
/// passing over each list once; with `many` false the keys are taken as
/// distinct, which spares a single removal the set.
fn unlist_all<K: std::hash::Hash + Eq + Copy>(
    map: &mut Index<K, usize>,
    keys: impl Iterator<Item = K>,
    gone: impl Fn(&usize) -> bool,
    many: bool,
) {
    if !many {
        for k in keys {
            unlist(map, &k, &gone);
        }
        return;
    }
    let mut seen = HashSet::with_hasher(FxBuildHasher);
    for k in keys {
        if seen.insert(k) {
            unlist(map, &k, &gone);
        }
    }
}

fn listed<K: std::hash::Hash + Eq, T: Copy>(map: &Index<K, T>, at: &K) -> Vec<T> {
    map.get(at).map_or_else(Vec::new, |l| l.as_slice().to_vec())
}

impl Pending {
    pub(crate) fn push(&mut self, d: Delivery) {
        let i = self.slots.len();
        list(&mut self.by_key, d.key, i);
        let made = self.by_maker.entry(d.maker).or_default();
        made.live += 1;
        made.list.push(i);
        list(&mut self.by_frame, frame_of(&d.key), i);
        list(&mut self.by_read, read_of(&d.key), i);
        list(&mut self.by_node, (d.key.node, d.key.generation), i);
        if awaits(&d) {
            *self.untaken.entry(frame_of(&d.key)).or_default() += 1;
        }
        if d.provenance.is_some() {
            self.traced += 1;
            let own = (d.key.cidx, d.key.node, d.key.generation);
            for &stand in std::iter::once(&own).chain(&d.below) {
                list(&mut self.by_stand, stand, i);
            }
            list(&mut self.by_frontier, frontier(&d), i);
        }
        let passed = d.passed;
        self.slots.push(Some(d));
        if passed {
            self.file_passed(i);
        }
        self.live += 1;
        #[cfg(all(test, not(loom)))]
        PEAK_LIVE.with(|p| p.set(p.get().max(self.live)));
    }

    pub(crate) fn remove(&mut self, i: usize) -> Delivery {
        let d = self.slots[i]
            .take()
            .expect("incr-core: a delivery is removed once");
        self.unfile_passed(i);
        self.unindex(std::slice::from_ref(&d));
        d
    }

    /// Removes deliveries `is`, in that order.
    fn remove_all(&mut self, is: Vec<usize>) -> Vec<Delivery> {
        let out: Vec<Delivery> = is
            .into_iter()
            .map(|i| {
                self.unfile_passed(i);
                self.slots[i]
                    .take()
                    .expect("incr-core: a delivery is removed once")
            })
            .collect();
        self.unindex(&out);
        out
    }

    /// Drops `out`, deliveries just taken from their slots, from every
    /// index, passing over each list they share once rather than once per
    /// delivery: a wide frame's deliveries share its frame's, maker's and
    /// node's lists.
    fn unindex(&mut self, out: &[Delivery]) {
        if out.is_empty() {
            return;
        }
        let many = out.len() > 1;
        let slots = &self.slots;
        let gone = |&j: &usize| slots[j].is_none();
        unlist_all(&mut self.by_key, out.iter().map(|d| d.key), gone, many);
        unlist_all(
            &mut self.by_frame,
            out.iter().map(|d| frame_of(&d.key)),
            gone,
            many,
        );
        unlist_all(
            &mut self.by_read,
            out.iter().map(|d| read_of(&d.key)),
            gone,
            many,
        );
        unlist_all(
            &mut self.by_node,
            out.iter().map(|d| (d.key.node, d.key.generation)),
            gone,
            many,
        );
        let traced = || out.iter().filter(|d| d.provenance.is_some());
        if traced().next().is_some() {
            self.traced -= traced().count();
            let stands = traced().flat_map(|d| {
                std::iter::once((d.key.cidx, d.key.node, d.key.generation))
                    .chain(d.below.iter().copied())
            });
            unlist_all(&mut self.by_stand, stands, gone, many);
            unlist_all(&mut self.by_frontier, traced().map(frontier), gone, many);
        }
        for d in out {
            self.unmake(d.maker);
            if awaits(d) {
                self.count_taken(frame_of(&d.key));
            }
        }
        self.live -= out.len();
        // No index is held across a point where none is live.
        if self.live == 0 {
            debug_assert!(
                self.by_maker.is_empty(),
                "a maker list outlived its deliveries"
            );
            self.slots.clear();
        }
    }

    fn unmake(&mut self, maker: u32) {
        let slots = &self.slots;
        unmade(&mut self.by_maker, maker, |&j| slots[j].is_some());
    }

    pub(crate) fn get(&self, i: usize) -> &Delivery {
        self.slots[i]
            .as_ref()
            .expect("incr-core: a delivery's index is live")
    }

    pub(crate) fn get_mut(&mut self, i: usize) -> &mut Delivery {
        self.slots[i]
            .as_mut()
            .expect("incr-core: a delivery's index is live")
    }

    /// Delivery `i`'s frame has read past it.
    pub(crate) fn pass(&mut self, i: usize) {
        let d = self.get_mut(i);
        if !d.passed {
            d.passed = true;
            self.file_passed(i);
        }
    }

    /// Delivery `i`'s latest take raised a copy of its payload keyed `key`.
    pub(crate) fn resume(&mut self, i: usize, key: PayloadKey) {
        let old = self.get_mut(i).resumed.replace(key);
        // Only a shallow filing is keyed by the copy's payload.
        if let Some(Filed::Shallow(above, payloads)) = self.filed.get_mut(&i) {
            // A copy keyed as the source shares the source's entry.
            if let Some(old) = old.filter(|&old| Some(old) != payloads[0]) {
                self.by_payload.remove(&(*above, old, i));
            }
            self.by_payload.insert((*above, key, i));
            payloads[1] = Some(key);
        }
    }

    fn file_passed(&mut self, i: usize) {
        let d = self.get(i);
        let Delivered::Unwind { nodes, .. } = &d.what else {
            return;
        };
        let above = d.key.cidx + 1;
        let filed = match nodes.deepest() {
            None => return,
            Some(0) => {
                let payloads = [nodes.source, d.resumed];
                for p in payloads.into_iter().flatten() {
                    self.by_payload.insert((above, p, i));
                }
                Filed::Shallow(above, payloads)
            }
            Some(depth) => {
                self.by_top.insert((above + depth, i));
                Filed::Deep(above + depth)
            }
        };
        self.filed.insert(i, filed);
    }

    fn unfile_passed(&mut self, i: usize) {
        match self.filed.remove(&i) {
            Some(Filed::Deep(top)) => {
                self.by_top.remove(&(top, i));
            }
            Some(Filed::Shallow(above, payloads)) => {
                for p in payloads.into_iter().flatten() {
                    self.by_payload.remove(&(above, p, i));
                }
            }
            None => {}
        }
    }

    /// Applies `ev` to the landings the passed deliveries carry: those of
    /// every passed delivery for a node's landing that a copy may hold,
    /// else only those of deliveries with a landing the event can reach.
    fn retire(&mut self, ev: &Retire, all: bool) {
        let apply = |d: &mut Delivery| {
            note_work(1);
            if d.key.cidx < ev.cidx() || all {
                let resumed = d.resumed;
                if let Delivered::Unwind { nodes, .. } = &mut d.what {
                    nodes.retire(d.key.cidx, ev, resumed);
                }
            }
        };
        let hit: Vec<usize> = if all {
            for d in self.slots.iter_mut().flatten().filter(|d| d.passed) {
                apply(d);
            }
            self.filed.keys().copied().collect()
        } else {
            // Short of a copied node's landing, an event retires only
            // landings of a frame below it that are above its position, or
            // at it for a containment of the delivery's own payload, so a
            // delivery none of whose landings reaches that high keeps them
            // all.
            let (from, contained) = match *ev {
                Retire::Above(at) | Retire::Landing { cidx: at, .. } => {
                    (at.saturating_add(1), None)
                }
                Retire::Contained { cidx: at, payload } => (at, Some((at, payload))),
            };
            let deep = self.by_top.range((from, 0)..).map(|&(_, i)| i);
            let shallow = contained.into_iter().flat_map(|(at, p)| {
                (self.by_payload.range((at, p, 0)..=(at, p, usize::MAX))).map(|&(_, _, i)| i)
            });
            let hit: Vec<usize> = deep.chain(shallow).collect();
            #[cfg(all(test, not(loom)))]
            self.check_unhit(ev, &hit);
            hit
        };
        for i in hit {
            self.unfile_passed(i);
            if !all {
                apply(self.get_mut(i));
            }
            self.file_passed(i);
        }
    }

    /// Checks that `ev` would leave as they are the landings of every
    /// passed delivery `retire` does not visit, on a list small enough to
    /// check whole.
    #[cfg(all(test, not(loom)))]
    fn check_unhit(&self, ev: &Retire, hit: &[usize]) {
        if self.slots.len() > 64 {
            return;
        }
        without_note_work(|| {
            for (i, d) in self.slots.iter().enumerate() {
                let reached = |d: &&Delivery| d.passed && d.key.cidx < ev.cidx();
                let Some(d) = d.as_ref().filter(reached).filter(|_| !hit.contains(&i)) else {
                    continue;
                };
                let Delivered::Unwind { nodes, .. } = &d.what else {
                    continue;
                };
                let mut retired = nodes.clone();
                retired.retire(d.key.cidx, ev, d.resumed);
                assert!(
                    nodes.clone().into_nodes() == retired.into_nodes(),
                    "incr-core: an event at {} skipped delivery {i}",
                    ev.cidx()
                );
            }
        });
    }

    /// The deliveries for read `key`, oldest first.
    pub(crate) fn keyed(&self, key: &ReadKey) -> &[usize] {
        self.by_key.get(key).map_or(&[], Few::as_slice)
    }

    /// The oldest delivery for read `key` that `f` accepts.
    pub(crate) fn find_keyed(&self, key: &ReadKey, f: impl Fn(&Delivery) -> bool) -> Option<usize> {
        self.keyed(key).iter().copied().find(|&i| f(self.get(i)))
    }

    /// Whether the frame of `node` at `cidx`, as `(node, generation, cidx)`,
    /// has a delivery for a read of its own runtime's node still to take.
    pub(crate) fn frame_awaits(&self, frame: (u32, u32, u32)) -> bool {
        let awaited = self.untaken.contains_key(&frame);
        debug_assert_eq!(
            awaited,
            listed(&self.by_frame, &frame)
                .into_iter()
                .any(|i| awaits(self.get(i))),
            "incr-core: a frame's untaken count matches its deliveries"
        );
        awaited
    }

    /// Whether that frame's read of `reading` has a delivery still to take:
    /// under `ordinal`, if given, and under `parts`, the frame's path,
    /// episode and run seq, if given.
    pub(crate) fn read_awaits(
        &self,
        frame: (u32, u32, u32),
        reading: u64,
        ordinal: Option<u32>,
        parts: Option<(u64, Option<u64>, u32)>,
    ) -> bool {
        let (node, generation, cidx) = frame;
        self.by_read
            .get(&(node, generation, cidx, reading))
            .is_some_and(|list| {
                list.as_slice().iter().any(|&i| {
                    let d = self.get(i);
                    awaits(d)
                        && ordinal.is_none_or(|o| d.key.ordinal == o)
                        && parts.is_none_or(|p| (d.key.path, d.key.episode, d.key.seq) == p)
                })
            })
    }

    /// Marks every delivery for read `key` taken.
    pub(crate) fn take_keyed(&mut self, key: &ReadKey) {
        for i in self.keyed(key).to_vec() {
            self.set_taken(i, true);
        }
    }

    /// Sets delivery `i`'s `taken`, which only this writes.
    pub(crate) fn set_taken(&mut self, i: usize, taken: bool) {
        let d = self.get_mut(i);
        if d.taken == taken {
            return;
        }
        let (counted, frame) = (d.key.foreign.is_none(), frame_of(&d.key));
        d.taken = taken;
        if counted && taken {
            self.count_taken(frame);
        } else if counted {
            *self.untaken.entry(frame).or_default() += 1;
        }
    }

    fn count_taken(&mut self, frame: (u32, u32, u32)) {
        let MapEntry::Occupied(mut e) = self.untaken.entry(frame) else {
            debug_assert!(false, "a frame's untaken count went below zero");
            return;
        };
        *e.get_mut() -= 1;
        if *e.get() == 0 {
            e.remove();
        }
    }

    /// Removes the walk directions for `key`, which hold no user value
    /// and so may drop inside the contexts' borrow.
    pub(crate) fn remove_walks(&mut self, key: &ReadKey) {
        drop(self.remove_keyed(key, |d| matches!(d.what, Delivered::Walk(..))));
    }

    /// Removes the deliveries for `key` that `f` picks and returns them:
    /// one may hold a user value, whose `Drop` may read and so must run
    /// outside the contexts' borrow.
    #[must_use]
    pub(crate) fn remove_keyed(
        &mut self,
        key: &ReadKey,
        f: impl Fn(&Delivery) -> bool,
    ) -> Vec<Delivery> {
        let mut removed = Vec::new();
        for i in self.keyed(key).to_vec() {
            if f(self.get(i)) {
                removed.push(self.remove(i));
            }
        }
        removed
    }

    /// The deliveries the run at `level` made, oldest first.
    pub(crate) fn of_maker(&self, level: u32) -> Vec<usize> {
        self.by_maker.get(&level).map_or_else(Vec::new, |made| {
            note_work(made.list.as_slice().len());
            made.list
                .as_slice()
                .iter()
                .copied()
                .filter(|&j| self.slots[j].is_some())
                .collect()
        })
    }

    /// Removes the deliveries the run at `level` made, oldest first.
    pub(crate) fn extract_maker(&mut self, level: u32) -> Vec<Delivery> {
        let is = self.of_maker(level);
        self.remove_all(is)
    }

    /// The deliveries for reads by the frame of `node` at `generation` and
    /// `cidx`, oldest first.
    pub(crate) fn of_frame(&self, node: u32, generation: u32, cidx: u32) -> Vec<usize> {
        listed(&self.by_frame, &(node, generation, cidx))
    }

    /// Removes the deliveries for reads by that frame, oldest first.
    pub(crate) fn extract_frame(&mut self, node: u32, generation: u32, cidx: u32) -> Vec<Delivery> {
        let is = self.of_frame(node, generation, cidx);
        self.remove_all(is)
    }

    /// Whether a frame of `node` at `generation`, at any position, has an
    /// outcome for one of its own runtime's reads still untaken. Its retry
    /// may be yet to land the node as the unbounded run's frame did; a
    /// reset of its maker's deliveries also leaves one after it has.
    pub(crate) fn owes_outcome(&self, node: u32, generation: u32) -> bool {
        self.by_node.get(&(node, generation)).is_some_and(|list| {
            list.as_slice().iter().any(|&i| {
                let d = self.get(i);
                !d.taken
                    && d.key.foreign.is_none()
                    && matches!(d.what, Delivered::Landed(..) | Delivered::Unwind { .. })
            })
        })
    }

    /// Removes, oldest first, the deliveries for reads by the frames of
    /// `node` at `generation`, at any position.
    pub(crate) fn extract_node(&mut self, node: u32, generation: u32) -> Vec<Delivery> {
        let is = listed(&self.by_node, &(node, generation));
        self.remove_all(is)
    }

    /// Whether any delivery carries a provenance record.
    pub(crate) fn has_traced(&self) -> bool {
        self.traced != 0
    }

    /// The deliveries with a record that stand for a read of `frame`'s
    /// node, as `(cidx, node, generation)`, oldest first.
    pub(crate) fn standing_for(&self, frame: (u32, u32, u32)) -> Vec<usize> {
        let mut found = listed(&self.by_stand, &frame);
        note_work(found.len());
        found.sort_unstable();
        found
    }

    /// The virtual frame `v`, as `(cidx, node, generation)`, popped: the
    /// deliveries with a record whose lowest frame sat just above it now
    /// stand for its read too.
    pub(crate) fn extend_below(&mut self, v: (u32, u32, u32)) {
        let Some(above) = self.by_frontier.remove(&(v.0 + 1)) else {
            return;
        };
        note_work(above.as_slice().len());
        for &i in above.as_slice() {
            self.get_mut(i).below.push(v);
            list(&mut self.by_stand, v, i);
            list(&mut self.by_frontier, v.0, i);
        }
    }

    pub(crate) fn into_vec(self) -> Vec<Delivery> {
        self.slots.into_iter().flatten().collect()
    }

    #[cfg(all(test, not(loom)))]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Delivery> {
        self.slots.iter().flatten()
    }
}

/// A virtual frame's blocked read that ran its node's compute, a virtual
/// frame above the frame's, which the retry runs again as that compute: a
/// walk would first visit deps the replay of that frame's run has left
/// Failed or Dirty (see [`Context::deliver_direct`]). It stays, taken or
/// not, until its maker's run ends, as a [`Delivery`] does. It keeps its
/// [`ReadKey`] less the position, which is where [`Directs`] holds it, and
/// the foreign runtime, which a blocked read has none of. Its [`RunSlot`]'s
/// path is the one through the read's compute (`run_path`), and its counts
/// are the runs of the node read, other nodes' in [`Directs::more_runs`].
/// A deep ring keeps one per virtual frame, so it is packed: 56 bytes.
#[derive(Clone, Copy)]
struct Direct {
    reading: u64,
    path: u64,
    /// The key's episode when `owner` has [`HAS_EPISODE`].
    episode: u64,
    /// [`EMPTY`] in a position holding none.
    node: u32,
    generation: u32,
    ordinal: u32,
    seq: u32,
    maker: u32,
    /// The run slot's owner, with [`TAKEN`], [`HAS_EPISODE`] and
    /// [`MORE_RUNS`].
    owner: u32,
    /// When it was pushed, against [`Directs::released`].
    stamp: u32,
    /// The runs of the node read along the run path.
    runs: u32,
}

const _: () = assert!(std::mem::size_of::<Direct>() == 56);

/// Runs along a path as `(node, generation, count)`.
type RunCounts = Few<(u32, u32, u32)>;

const EMPTY: u32 = u32::MAX;
const TAKEN: u32 = 1 << 31;
const HAS_EPISODE: u32 = 1 << 30;
const MORE_RUNS: u32 = 1 << 29;
const OWNER: u32 = MORE_RUNS - 1;

const VACANT: Direct = Direct {
    reading: 0,
    path: 0,
    episode: 0,
    node: EMPTY,
    generation: 0,
    ordinal: 0,
    seq: 0,
    maker: 0,
    owner: 0,
    stamp: 0,
    runs: 0,
};

const CHUNK: usize = 4096;

impl Direct {
    fn new(key: &ReadKey, maker: u32, owner: u32, stamp: u32) -> Self {
        debug_assert!(owner <= OWNER, "a run slot's owner is a depth");
        Self {
            reading: key.reading,
            path: key.path,
            episode: key.episode.unwrap_or(0),
            node: key.node,
            generation: key.generation,
            ordinal: key.ordinal,
            seq: key.seq,
            maker,
            owner: owner
                | if key.episode.is_some() {
                    HAS_EPISODE
                } else {
                    0
                },
            stamp,
            runs: 0,
        }
    }

    fn episode(&self) -> Option<u64> {
        (self.owner & HAS_EPISODE != 0).then_some(self.episode)
    }

    fn taken(&self) -> bool {
        self.owner & TAKEN != 0
    }

    fn set_taken(&mut self, taken: bool) {
        if taken {
            self.owner |= TAKEN;
        } else {
            self.owner &= !TAKEN;
        }
    }

    /// The node read, as `(slot, generation)`.
    fn read_node(&self) -> (u32, u32) {
        (self.reading as u32, (self.reading >> 32) as u32)
    }

    /// The path of the read's compute, the frame above this one, whose
    /// runs it counts: [`run_snapshot`]'s.
    fn run_path(&self) -> u64 {
        extend_path(
            self.path,
            Step {
                node: self.node,
                episode: self.episode(),
                reading: self.reading,
                ordinal: self.ordinal,
                seq: self.seq,
            },
        )
    }

    fn is(&self, key: &ReadKey) -> bool {
        (self.node, self.generation, self.reading, self.ordinal)
            == (key.node, key.generation, key.reading, key.ordinal)
            && (self.path, self.seq, self.episode()) == (key.path, key.seq, key.episode)
            && key.foreign.is_none()
    }

    fn of(&self, node: u32, generation: u32) -> bool {
        (self.node, self.generation) == (node, generation)
    }
}

/// A context's direct deliveries, at most one per read, by their frame's
/// position less the context's boundary. A deep chain's retries leave one
/// per virtual frame until their makers' runs end, so they keep no
/// [`Pending`] slot or index of their own, and a position holds its first
/// in place, any further ones in `more_at`.
#[derive(Default)]
pub(crate) struct Directs {
    boundary: u32,
    /// In chunks of [`CHUNK`] positions, as many as the highest position the
    /// context has used, empty or not, so a deep direct that goes and comes
    /// back finds its position made, and growth copies at most a chunk.
    at: Vec<Vec<Direct>>,
    more_at: HashMap<u32, Vec<Direct>, FxBuildHasher>,
    /// Counts of runs of nodes other than the one read, by position and
    /// stamp, for directs with [`MORE_RUNS`].
    more_runs: HashMap<(u32, u32), RunCounts, FxBuildHasher>,
    live: usize,
    next_stamp: u32,
    /// Nodes whose frames a user panic's release ended, with the stamp
    /// their directs then reached: those pushed before it are gone, and
    /// a removal that reaches them takes them out. Cold: only a release
    /// fills it, and it empties with the directs.
    released: HashMap<(u32, u32), u32, FxBuildHasher>,
    /// Positions by maker, dropped lazily as in [`Pending::by_maker`].
    by_maker: HashMap<u32, Made, FxBuildHasher>,
}

impl Directs {
    fn new(boundary: u32) -> Self {
        Self {
            boundary,
            ..Self::default()
        }
    }

    /// Where the frame at `cidx` keeps its directs: none below the
    /// context's boundary.
    fn position(&self, cidx: u32) -> Option<usize> {
        cidx.checked_sub(self.boundary).map(|i| i as usize)
    }

    fn slot(&self, i: usize) -> Option<&Direct> {
        self.at.get(i / CHUNK).and_then(|c| c.get(i % CHUNK))
    }

    fn slot_mut(&mut self, i: usize) -> Option<&mut Direct> {
        self.at
            .get_mut(i / CHUNK)
            .and_then(|c| c.get_mut(i % CHUNK))
    }

    fn dead(&self, d: &Direct) -> bool {
        !self.released.is_empty()
            && (self.released.get(&(d.node, d.generation))).is_some_and(|&s| d.stamp < s)
    }

    /// The directs at position `i`, gone ones included.
    fn all_at(&self, i: usize) -> impl Iterator<Item = &Direct> {
        let first = self.slot(i).filter(|d| d.node != EMPTY);
        let more = (first.is_some() && !self.more_at.is_empty())
            .then(|| self.more_at.get(&(i as u32)))
            .flatten()
            .map_or(&[][..], Vec::as_slice);
        first.into_iter().chain(more)
    }

    /// The first live direct of the frame at `cidx` that `pick` accepts.
    fn find_at(&self, cidx: u32, pick: impl Fn(&Direct) -> bool) -> Option<&Direct> {
        let i = self.position(cidx)?;
        let first = self.slot(i).filter(|d| d.node != EMPTY)?;
        let found = |d: &&Direct| pick(d) && !self.dead(d);
        if found(&first) {
            return Some(first);
        }
        if self.more_at.is_empty() {
            return None;
        }
        self.more_at.get(&(i as u32))?.iter().find(found)
    }

    /// Applies `f` to each direct of the frame at `cidx` that `pick`
    /// accepts, stopping once `f` returns true.
    fn find_mut(
        &mut self,
        cidx: u32,
        pick: impl Fn(&Direct) -> bool,
        mut f: impl FnMut(&mut Direct, &mut RunCounts) -> bool,
    ) {
        let Some(i) = self.position(cidx) else {
            return;
        };
        let released = &self.released;
        let alive = |d: &Direct| {
            d.node != EMPTY
                && (released.is_empty()
                    || (released.get(&(d.node, d.generation))).is_none_or(|&s| d.stamp >= s))
                && pick(d)
        };
        let mut none = Few::None;
        let first = self
            .at
            .get_mut(i / CHUNK)
            .and_then(|c| c.get_mut(i % CHUNK));
        let Some(first) = first.filter(|d| d.node != EMPTY) else {
            return;
        };
        let more_runs = &mut self.more_runs;
        let mut visit = |d: &mut Direct| {
            if !alive(d) {
                return false;
            }
            if d.owner & MORE_RUNS != 0 {
                let runs = more_runs.entry((i as u32, d.stamp)).or_default();
                f(d, runs)
            } else {
                let stop = f(d, &mut none);
                if !matches!(none, Few::None) {
                    d.owner |= MORE_RUNS;
                    more_runs.insert((i as u32, d.stamp), std::mem::take(&mut none));
                }
                stop
            }
        };
        if visit(first) {
            return;
        }
        if let Some(more) = (!self.more_at.is_empty())
            .then(|| self.more_at.get_mut(&(i as u32)))
            .flatten()
        {
            for d in more {
                if visit(d) {
                    return;
                }
            }
        }
    }

    fn get(&self, key: &ReadKey) -> Option<&Direct> {
        self.find_at(key.cidx, |d| d.is(key))
    }

    fn runs_of(&self, i: usize, d: &Direct) -> RunSlot {
        let (node, generation) = d.read_node();
        let mut counts = Few::None;
        if d.runs != 0 {
            counts.push((node, generation, d.runs));
        }
        if d.owner & MORE_RUNS != 0 {
            if let Some(more) = self.more_runs.get(&(i as u32, d.stamp)) {
                for &c in more.as_slice() {
                    counts.push(c);
                }
            }
        }
        RunSlot {
            path: d.run_path(),
            owner: d.owner & OWNER,
            counts,
        }
    }

    fn push(&mut self, key: ReadKey, maker: u32, runs: RunSlot) {
        debug_assert!(key.foreign.is_none(), "a blocked read is its own runtime's");
        let i = self
            .position(key.cidx)
            .expect("incr-core: a direct read is its context's frame's");
        if self.next_stamp == u32::MAX {
            self.restamp();
        }
        let stamp = self.next_stamp;
        self.next_stamp += 1;
        let mut d = Direct::new(&key, maker, runs.owner, stamp);
        debug_assert!(
            runs.path == d.run_path(),
            "a direct's runs are along its read's compute"
        );
        let read = d.read_node();
        let mut more = Few::None;
        for &(n, g, k) in runs.counts.as_slice() {
            if (n, g) == read {
                d.runs = k;
            } else {
                more.push((n, g, k));
            }
        }
        if !matches!(more, Few::None) {
            d.owner |= MORE_RUNS;
            self.more_runs.insert((i as u32, stamp), more);
        }
        self.make(i);
        let first = &mut self.at[i / CHUNK][i % CHUNK];
        if first.node == EMPTY {
            *first = d;
        } else {
            self.more_at.entry(i as u32).or_default().push(d);
        }
        let made = self.by_maker.entry(maker).or_default();
        made.live += 1;
        made.list.push(i);
        self.live += 1;
        #[cfg(all(test, not(loom)))]
        PEAK_DIRECTS.with(|p| p.set(p.get().max(self.live)));
    }

    /// Makes position `i`: the chunks below it whole, its own to the next
    /// power of two, so a shallow context pays for the positions it uses.
    fn make(&mut self, i: usize) {
        let (c, j) = (i / CHUNK, i % CHUNK);
        let grow = |chunk: &mut Vec<Direct>, len: usize| {
            if chunk.len() < len {
                note_work(len - chunk.len());
                chunk.resize(len, VACANT);
            }
        };
        if let Some(last) = self.at.len().checked_sub(1).filter(|&l| l < c) {
            grow(&mut self.at[last], CHUNK);
        }
        while self.at.len() < c {
            let mut chunk = Vec::new();
            grow(&mut chunk, CHUNK);
            self.at.push(chunk);
        }
        if self.at.len() == c {
            self.at.push(Vec::new());
        }
        grow(&mut self.at[c], (j + 1).next_power_of_two().min(CHUNK));
    }

    /// The stamps ran out: the gone directs are taken out and the rest
    /// restamped from 0. Once per 2^32 pushes into one context.
    #[cold]
    #[inline(never)]
    fn restamp(&mut self) {
        let positions: Vec<usize> = (0..self.at.len() * CHUNK)
            .filter(|&i| self.all_at(i).any(|d| self.dead(d)))
            .collect();
        for i in positions {
            let released = std::mem::take(&mut self.released);
            self.remove_at(i, true, |d| {
                released
                    .get(&(d.node, d.generation))
                    .is_some_and(|&s| d.stamp < s)
            });
            self.released = released;
        }
        self.released.clear();
        let mut old = std::mem::take(&mut self.more_runs);
        let mut stamp = 0;
        let mut restamp = |i: u32, d: &mut Direct| {
            if d.owner & MORE_RUNS != 0 {
                let runs = old.remove(&(i, d.stamp)).unwrap_or_default();
                self.more_runs.insert((i, stamp), runs);
            }
            d.stamp = stamp;
            stamp += 1;
        };
        for (c, chunk) in self.at.iter_mut().enumerate() {
            for (j, d) in chunk.iter_mut().enumerate() {
                if d.node != EMPTY {
                    restamp((c * CHUNK + j) as u32, d);
                }
            }
        }
        for (&i, more) in self.more_at.iter_mut() {
            for d in more {
                restamp(i, d);
            }
        }
        debug_assert!(old.is_empty(), "more runs outlived their directs");
        self.next_stamp = stamp;
    }

    /// Removes the directs at position `i`, gone ones included, that `f`
    /// picks. A caller removing all of a maker's drops its list itself
    /// (`maker_listed` false).
    fn remove_at(&mut self, i: usize, maker_listed: bool, f: impl Fn(&Direct) -> bool) {
        let Some(first) = self.slot(i).copied().filter(|d| d.node != EMPTY) else {
            return;
        };
        let mut gone = Few::None;
        let mut more = if self.more_at.is_empty() {
            None
        } else {
            self.more_at.remove(&(i as u32))
        };
        if let Some(v) = more.as_mut() {
            v.retain(|d| {
                let out = f(d);
                if out {
                    gone.push(*d);
                }
                !out
            });
        }
        if f(&first) {
            gone.push(first);
            let next = match more.as_mut().filter(|v| !v.is_empty()) {
                Some(v) => v.remove(0),
                None => VACANT,
            };
            *self
                .slot_mut(i)
                .expect("a position holding directs is made") = next;
        }
        if let Some(v) = more.filter(|v| !v.is_empty()) {
            self.more_at.insert(i as u32, v);
        }
        let gone = gone.as_slice();
        self.live -= gone.len();
        for d in gone {
            if d.owner & MORE_RUNS != 0 {
                self.more_runs.remove(&(i as u32, d.stamp));
            }
            if maker_listed {
                let (at, more_at) = (&self.at, &self.more_at);
                let mut seen = HashSet::new();
                unmade(&mut self.by_maker, d.maker, |&j| {
                    holds_maker(at, more_at, j, d.maker) && seen.insert(j)
                });
            }
        }
        if self.live == 0 {
            self.released.clear();
            debug_assert!(
                self.by_maker.is_empty() && self.more_at.is_empty() && self.more_runs.is_empty(),
                "a direct list outlived its directs"
            );
        }
    }

    /// Sends the read `key` to its compute, unless a direct one already
    /// does.
    pub(crate) fn deliver(&mut self, key: ReadKey, maker: u32, runs: RunSlot) {
        if self.get(&key).is_none() {
            self.push(key, maker, runs);
        }
    }

    /// Sends it again with `runs`: one from `maker` is read again in place,
    /// one from another maker is replaced.
    pub(crate) fn redeliver(&mut self, key: ReadKey, maker: u32, runs: RunSlot) {
        match self.get(&key).map(|d| d.maker) {
            Some(m) if m == maker => self.find_mut(
                key.cidx,
                |d| d.is(&key),
                |d, more| {
                    debug_assert!(
                        runs.path == d.run_path(),
                        "a direct's runs are along its read's compute"
                    );
                    let read = d.read_node();
                    d.runs = 0;
                    *more = Few::None;
                    for &(n, g, k) in runs.counts.as_slice() {
                        if (n, g) == read {
                            d.runs = k;
                        } else {
                            more.push((n, g, k));
                        }
                    }
                    d.owner = (d.owner & (HAS_EPISODE | MORE_RUNS)) | runs.owner;
                    true
                },
            ),
            Some(_) => {
                if let Some(i) = self.position(key.cidx) {
                    self.remove_at(i, true, |d| d.is(&key));
                }
                self.push(key, maker, runs);
            }
            None => self.push(key, maker, runs),
        }
    }

    /// Takes the untaken direct for read `key`, returning the runs its
    /// retry reads from.
    pub(crate) fn take(&mut self, key: &ReadKey) -> Option<RunSlot> {
        let i = self.position(key.cidx)?;
        let mut found = None;
        self.find_mut(
            key.cidx,
            |d| d.is(key) && !d.taken(),
            |d, _| {
                d.set_taken(true);
                found = Some(*d);
                true
            },
        );
        found.map(|d| self.runs_of(i, &d))
    }

    /// Marks the direct for read `key` taken, if there is one.
    pub(crate) fn take_keyed(&mut self, key: &ReadKey) {
        self.find_mut(
            key.cidx,
            |d| d.is(key),
            |d, _| {
                d.set_taken(true);
                true
            },
        );
    }

    /// Counts the run of `node` at `generation` along `path` in the
    /// untaken direct for read `key` whose runs follow that path.
    pub(crate) fn bump(&mut self, key: &ReadKey, path: u64, node: u32, generation: u32) {
        self.find_mut(
            key.cidx,
            |d| d.is(key),
            |d, more| {
                if !d.taken() && d.run_path() == path {
                    if (node, generation) == d.read_node() {
                        d.runs += 1;
                    } else {
                        match (more.as_mut_slice().iter_mut())
                            .find(|(n, g, _)| (*n, *g) == (node, generation))
                        {
                            Some((_, _, k)) => *k += 1,
                            None => more.push((node, generation, 1)),
                        }
                    }
                }
                true
            },
        );
    }

    /// The positions listed for the run at `level`, each once, lowest
    /// first.
    fn of_maker(&self, level: u32) -> Vec<usize> {
        let mut at: Vec<usize> = self
            .by_maker
            .get(&level)
            .map_or_else(Vec::new, |m| m.list.as_slice().to_vec());
        note_work(at.len());
        at.sort_unstable();
        at.dedup();
        at
    }

    fn rearm_at(&mut self, i: usize, pick: impl Fn(&Direct) -> bool) {
        if let Some(d) = self.slot_mut(i).filter(|d| d.node != EMPTY && pick(d)) {
            d.set_taken(false);
        }
        if let Some(more) = (!self.more_at.is_empty())
            .then(|| self.more_at.get_mut(&(i as u32)))
            .flatten()
        {
            for d in more.iter_mut().filter(|d| pick(d)) {
                d.set_taken(false);
            }
        }
    }

    /// The run at `level` was doomed: its directs are read again.
    pub(crate) fn rearm_maker(&mut self, level: u32) {
        for i in self.of_maker(level) {
            self.rearm_at(i, |d| d.maker == level);
        }
    }

    /// The frame at `frame`, as `(node, generation, cidx)`, runs again:
    /// its directs are read again.
    pub(crate) fn rearm_frame(&mut self, (node, generation, cidx): (u32, u32, u32)) {
        if let Some(i) = self.position(cidx) {
            self.rearm_at(i, |d| d.of(node, generation));
        }
    }

    /// The run at `level` ended: its directs go.
    pub(crate) fn remove_maker(&mut self, level: u32) {
        let at = self.of_maker(level);
        self.by_maker.remove(&level);
        for i in at {
            self.remove_at(i, false, |d| d.maker == level);
        }
    }

    /// The directs for reads by the frame at `frame` go.
    pub(crate) fn remove_frame(&mut self, (node, generation, cidx): (u32, u32, u32)) {
        if let Some(i) = self.position(cidx) {
            self.remove_at(i, true, |d| d.of(node, generation));
        }
    }

    /// The directs for reads by the frames of `node` at `generation`, at
    /// any position, go: lazily, as their removals by maker or frame take
    /// them out, since a release is cold and the directs keep no index by
    /// node.
    pub(crate) fn remove_node(&mut self, node: u32, generation: u32) {
        if self.live != 0 {
            self.released.insert((node, generation), self.next_stamp);
        }
    }

    /// As [`Pending::frame_awaits`].
    pub(crate) fn frame_awaits(&self, (node, generation, cidx): (u32, u32, u32)) -> bool {
        (self.find_at(cidx, |d| !d.taken() && d.of(node, generation))).is_some()
    }

    /// As [`Pending::read_awaits`].
    pub(crate) fn read_awaits(
        &self,
        (node, generation, cidx): (u32, u32, u32),
        reading: u64,
        ordinal: Option<u32>,
        parts: Option<(u64, Option<u64>, u32)>,
    ) -> bool {
        let found = self.find_at(cidx, |d| {
            !d.taken()
                && d.of(node, generation)
                && d.reading == reading
                && ordinal.is_none_or(|o| d.ordinal == o)
                && parts.is_none_or(|p| (d.path, d.episode(), d.seq) == p)
        });
        found.is_some()
    }
}

fn holds_maker(
    at: &[Vec<Direct>],
    more_at: &HashMap<u32, Vec<Direct>, FxBuildHasher>,
    j: usize,
    maker: u32,
) -> bool {
    let Some(first) = at.get(j / CHUNK).and_then(|c| c.get(j % CHUNK)) else {
        return false;
    };
    first.node != EMPTY
        && (first.maker == maker
            || (more_at.get(&(j as u32))).is_some_and(|m| m.iter().any(|d| d.maker == maker)))
}

/// One of `k`'s entries went (see [`Made::shed`]).
fn unmade<K: std::hash::Hash + Eq>(
    index: &mut HashMap<K, Made, FxBuildHasher>,
    k: K,
    keep: impl FnMut(&usize) -> bool,
) {
    let MapEntry::Occupied(mut e) = index.entry(k) else {
        debug_assert!(false, "a maker's live count went below zero");
        return;
    };
    if e.get_mut().shed(keep) {
        e.remove();
    }
}

impl Delivery {
    /// An untaken, unblocked delivery of `what` for the read `key`.
    pub(crate) fn new(key: ReadKey, maker: u32, what: Delivered) -> Self {
        Self {
            key,
            maker,
            taken: false,
            blocked: false,
            what,
            provenance: None,
            below: Vec::new(),
            resumed: None,
            landed: None,
            passed: false,
            runs: RunSlot::default(),
        }
    }

    pub(crate) fn taken(&self) -> bool {
        self.taken
    }

    fn is_untaken_panic(&self) -> bool {
        !self.taken && matches!(self.what, Delivered::Unwind { cycle: false, .. })
    }
}

/// A delivery's hold on its runtime's provenance record, which stays
/// live while some delivery holds it.
pub(crate) trait Provenance {
    fn record(&self) -> u64;
    /// The read the record names was answered by taking its delivery.
    fn spend(&self);
}

/// This thread's Failed read that a provenance record names: the
/// runtime, the record and the node read.
#[derive(Clone, Copy)]
pub(crate) struct Carry {
    pub(crate) rt: RuntimeId,
    pub(crate) record: u64,
    pub(crate) node: u32,
}

pub(crate) enum Delivered {
    /// An unwind out of a non-speculative entry: a cycle unwind, whose
    /// payload is the `CycleUnwind`, or a user panic. `poison` is the flag
    /// it raises on the reader (`Runtime::unwound_poison`), 0 for the
    /// cycle of a node that landed `Cyclic`, which a read of the node
    /// raises with the node landed.
    Unwind {
        cycle: bool,
        poison: u16,
        /// A member was marked failed in the run the panic came out of,
        /// so a member run that takes it again is not.
        attributed: bool,
        payload: Box<dyn Any + Send>,
        /// The landings the panic made that a walk's entry may contain
        /// with it (`runtime_provenance`), which follow each copy taken.
        nodes: Landings,
    },
    /// The entry landed without a Clean value. The unbounded run's read
    /// takes the slot as the compute left it, raising the flag beside it
    /// on the reader (`Runtime::dirty_read_poison`), where a later read
    /// would walk and run the node again unless it holds a memo. It is
    /// that value, cloned when the delivery is made, because a later run
    /// in the context may compute the node again before the retry reads
    /// it. Until a read takes it, it is a clone the unbounded run never
    /// builds.
    Landed(Unbuilt<dyn Any + Send>, u16),
    /// The rest of the walk of the frame's blocked read, from the compute
    /// a doom stopped it at: the retry's walk of that read resumes it, as
    /// the unbounded run's walk goes on after that compute. With the node
    /// of the entry the stop pushed back, which the context's run of that
    /// compute pops: a doomed entry is not pushed back, so a pending entry
    /// of its node on top is a repeat the walk still owes.
    Walk(Box<crate::runtime::Walk>, Option<u32>),
}

pub(crate) struct Context {
    pub(crate) id: u64,
    pub(crate) rt: RuntimeId,
    /// The position of level 0.
    pub(crate) boundary: u32,
    pub(crate) levels: Vec<VecDeque<Entry>>,
    /// How many entries of each node and generation `levels` holds, built
    /// when [`Self::queues`] first asks and kept in step from then on. Only
    /// a `Shared` reservation's end asks, so other contexts count nothing,
    /// and a scan per ask instead is quadratic on a long budget-1 chain.
    queued: Option<HashMap<(u32, u32), u32, FxBuildHasher>>,
    pub(crate) vstack: Vec<VFrame>,
    /// Each node's `vstack` positions, lowest first, with the generation
    /// its virtual frame recorded.
    pub(crate) vindex: HashMap<u32, Vec<(u32, u32)>, FxBuildHasher>,
    /// The level of the entry the driver is running, or `None` until the
    /// loop starts.
    pub(crate) running: Option<u32>,
    pub(crate) pending: Pending,
    pub(crate) directs: Directs,
    pub(crate) running_entry: Option<Entry>,
    /// The running entry still fronts its level: the driver has neither
    /// popped nor truncated it.
    running_fronts: bool,
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
    /// The walk a `Walk` delivery just taken resumes, with the read it is
    /// for and the reading frame's position, until that read's walk takes
    /// it or the read ends.
    pub(crate) resume: Option<(u64, u32, crate::runtime::Walk)>,
    /// The node, generation and reading frame's position of the read a
    /// `Direct` delivery just taken sends to its compute, until that
    /// read's walk does or the read ends.
    pub(crate) direct: Option<(u32, u32, u32)>,
    /// The lowlink of each popped virtual frame whose run is not over, as
    /// `(node, generation, cidx, path, low)`, until the run that finishes
    /// it starts there: the deferrals that lowered it ran in the doomed
    /// runs, and a retry that resumes their walk or takes a delivery does
    /// not meet them again.
    pub(crate) lows: Vec<(u32, u32, u32, u64, u32)>,
    /// The walked watermark of each doomed frame's earliest claim, as
    /// `(node, generation, cidx, path, maker, rev)`, until the run that
    /// finishes the frame starts there or the maker's run ends: the
    /// unbounded run's frame claimed once, before any set its doomed runs
    /// made.
    pub(crate) claim_revs: EarliestByFrame,
    /// The thread's failure count when each doomed member run's earliest
    /// run began, kept as `claim_revs` is: the unbounded run's run of the
    /// member nests every failure marked since, whichever step of the
    /// replay marked it.
    pub(crate) run_failures: EarliestByFrame,
    /// The count slots of the reads at positions above `boundary`. A run's
    /// seq is its node's count in its slot when it starts (see
    /// [`start_run`]), which each undoomed end of a run and each rerun of a
    /// dissolved or stale root raises, so a second run of a node at one
    /// position reads none of the first's deliveries. A read sets its slot
    /// as it starts, empty or to what the delivery it resumes kept, so a
    /// retry counts as the run it redoes did.
    pub(crate) runs: RunSlots,
    /// The level of a landed point a read's judgement met above the
    /// position it read, until the driver drops the chain above it.
    pub(crate) drop_at: Option<u32>,
    /// The level a read in the running entry's run dropped the chain from
    /// cooperatively, by dooming every frame of the run: the driver drops
    /// it once the run comes back discarded.
    pub(crate) drop_point: Option<u32>,
    /// The provenance record that the delivery for this read takes,
    /// published before the frame above it landed Failed.
    pub(crate) provenance: Option<(ReadKey, Rc<dyn Provenance>)>,
    pub(crate) frame_keys: FrameKeys,
    /// Ends the reservations of this context's dooms when it is removed.
    _reserved: crate::reserve::Ends,
}

/// The parts of a read key for a frame of a context's runtime that only
/// that runtime can give, for a read of another runtime's node by the
/// frame: the frame's node generation, its path and its member episode.
#[derive(Clone, Copy)]
pub(crate) struct FrameKeys {
    /// The runtime that opened the context, live while the context is open.
    pub(crate) owner: *const (),
    /// `(owner, node, cidx, member)` to the frame's key parts.
    pub(crate) parts: unsafe fn(*const (), u32, u32, bool) -> KeyParts,
}

/// A frame's node generation, path, member episode and run's seq (see
/// [`Context::runs`]).
pub(crate) type KeyParts = (u32, u64, Option<u64>, u32);

impl Context {
    pub(crate) fn new(
        id: u64,
        rt: RuntimeId,
        boundary: u32,
        target: Entry,
        frame_keys: FrameKeys,
    ) -> Self {
        Self {
            id,
            rt,
            boundary,
            levels: vec![VecDeque::from([target])],
            queued: None,
            vstack: Vec::new(),
            vindex: HashMap::default(),
            running: None,
            running_entry: None,
            running_fronts: false,
            pending: Pending::default(),
            directs: Directs::new(boundary),
            spec_base: target.spec,
            claims: Vec::new(),
            episodes: Vec::new(),
            memo: Vec::new(),
            resume: None,
            direct: None,
            lows: Vec::new(),
            claim_revs: EarliestByFrame::default(),
            run_failures: EarliestByFrame::default(),
            runs: RunSlots::default(),
            drop_at: None,
            drop_point: None,
            provenance: None,
            frame_keys,
            _reserved: crate::reserve::Ends(id),
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

    /// Marks `entry`, the front of `level`, as the one the driver runs.
    pub(crate) fn run_entry(&mut self, level: u32, entry: Entry) {
        self.running = Some(level);
        self.running_entry = Some(entry);
        self.running_fronts = true;
    }

    /// Whether an entry of `node` at `generation` has yet to run: the
    /// running entry is not one until the driver pops it.
    pub(crate) fn queues(&mut self, node: u32, generation: u32) -> bool {
        let running = self.running_fronts
            && self
                .running_entry
                .is_some_and(|e| e.node == node && e.generation == generation);
        let queued = self
            .queued
            .get_or_insert_with(|| Self::count_levels(&self.levels));
        queued.get(&(node, generation)).copied().unwrap_or(0) > u32::from(running)
    }

    #[cold]
    #[inline(never)]
    fn count_levels(levels: &[VecDeque<Entry>]) -> HashMap<(u32, u32), u32, FxBuildHasher> {
        let mut queued = HashMap::default();
        for e in levels.iter().flatten() {
            *queued.entry((e.node, e.generation)).or_default() += 1;
        }
        queued
    }

    /// Whether `node`'s virtual frame at `generation` stands for a frame
    /// on the unbounded run's stack ([`Self::stacked`]).
    pub(crate) fn stacks(&self, node: u32, generation: u32) -> bool {
        self.position(node, generation)
            .is_some_and(|p| (p as usize) < self.stacked().len())
    }

    pub(crate) fn push_entry(&mut self, level: u32, entry: Entry) {
        let level = level as usize;
        if self.levels.len() <= level {
            self.levels.resize_with(level + 1, VecDeque::new);
        }
        self.levels[level].push_back(entry);
        if self.queued.is_some() {
            self.count_in(entry);
        }
    }

    /// Removes the oldest entry at `level`, the one the driver ran.
    pub(crate) fn pop_entry(&mut self, level: u32) {
        self.drop_oldest(level);
        self.trim_levels();
    }

    /// Removes the oldest entry at `level`, leaving empty levels in place.
    pub(crate) fn drop_oldest(&mut self, level: u32) {
        if let Some(e) = self.levels[level as usize].pop_front() {
            if self.queued.is_some() {
                self.count_out(e);
            }
            if self.running == Some(level) {
                self.running_fronts = false;
            }
        }
    }

    /// Drops every level from `len` up with its entries.
    pub(crate) fn truncate_levels(&mut self, len: usize) {
        if self.queued.is_some() {
            for q in self.levels.split_off(len.min(self.levels.len())) {
                for e in q {
                    self.count_out(e);
                }
            }
        } else {
            self.levels.truncate(len);
        }
        if self.running.is_some_and(|level| level as usize >= len) {
            self.running_fronts = false;
        }
    }

    #[cold]
    #[inline(never)]
    fn count_in(&mut self, e: Entry) {
        if let Some(q) = &mut self.queued {
            *q.entry((e.node, e.generation)).or_default() += 1;
        }
    }

    #[cold]
    #[inline(never)]
    fn count_out(&mut self, e: Entry) {
        let Some(q) = &mut self.queued else {
            return;
        };
        if let MapEntry::Occupied(mut n) = q.entry((e.node, e.generation)) {
            *n.get_mut() -= 1;
            if *n.get() == 0 {
                n.remove();
            }
        }
    }

    /// Drops the empty levels on top, so `next_entry` finds the highest
    /// entry without walking the empty levels a push above budget 1
    /// leaves below it.
    pub(crate) fn trim_levels(&mut self) {
        while self.levels.len() > 1 && self.levels.last().is_some_and(VecDeque::is_empty) {
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
        let stack = self.vindex.entry(frame.node).or_default();
        #[cfg(all(test, not(loom)))]
        if !stack.is_empty() {
            VINDEX_REPUSHES.with(|n| n.set(n.get() + 1));
        }
        stack.push((pos, frame.generation));
        self.vstack.push(frame);
        #[cfg(all(test, not(loom)))]
        PEAK_VSTACK.with(|p| p.set(p.get().max(self.vstack.len())));
    }

    /// The highest virtual frame below position `cidx`.
    pub(crate) fn vframe_below(&self, cidx: u32) -> Option<&VFrame> {
        let above = cidx.saturating_sub(self.boundary) as usize;
        self.vstack[..above.min(self.vstack.len())].last()
    }

    /// Delivers `what` to the read virtual frame `below` was blocked in.
    #[must_use]
    pub(crate) fn deliver_blocked(&mut self, below: &VFrame, what: Delivered) -> Vec<Delivery> {
        self.push_blocked(below, what, None)
    }

    /// Delivers the unwind out of the running frame at `cidx` to the
    /// read of [`Self::unwind_target`], with the provenance record
    /// published for that read.
    #[must_use]
    pub(crate) fn deliver_unwind(&mut self, cidx: u32, what: Delivered) -> Vec<Delivery> {
        let below = self
            .unwind_target(cidx)
            .expect("incr-core: an unwind is delivered to a virtual frame")
            .clone();
        let key = ReadKey::blocked(&below);
        if let Delivered::Unwind { payload, .. } = &what {
            let reader = (below.node, below.generation, below.path, below.seq);
            note_took(self.rt, reader, below.reading, &**payload);
        }
        let provenance = self
            .provenance
            .take()
            .and_then(|(k, p)| (k == key).then_some(p));
        self.push_blocked(&below, what, provenance)
    }

    fn push_blocked(
        &mut self,
        below: &VFrame,
        what: Delivered,
        provenance: Option<Rc<dyn Provenance>>,
    ) -> Vec<Delivery> {
        let key = ReadKey::blocked(below);
        let replaced = self.pending.remove_keyed(&key, |d| d.blocked);
        let mut d = Delivery::new(key, below.maker, what);
        d.blocked = true;
        d.provenance = provenance;
        self.pending.push(d);
        replaced
    }

    /// The virtual frame the driver delivers an unwind out of the frame at
    /// `cidx` to, when that is the running frame: the running entry's, or
    /// the one a pop finishes.
    pub(crate) fn unwind_target(&self, cidx: u32) -> Option<&VFrame> {
        let level = self.running?;
        if self.boundary + level != cidx {
            return None;
        }
        self.vframe_at(cidx.checked_sub(1)?)
    }

    /// The read of [`Self::unwind_target`] for a landing of `node` at
    /// `cidx`, when that read is of `node`.
    pub(crate) fn landing_reader(&self, node: u32, generation: u32, cidx: u32) -> Option<ReadKey> {
        let below = self.unwind_target(cidx)?;
        (below.reading == u64::from(node) | (u64::from(generation) << 32))
            .then(|| ReadKey::blocked(below))
    }

    /// The virtual frame at position `cidx`.
    pub(crate) fn vframe_at(&self, cidx: u32) -> Option<&VFrame> {
        let v = self.vstack.get(cidx.checked_sub(self.boundary)? as usize)?;
        (v.cidx == cidx).then_some(v)
    }

    /// The untaken user-panic deliveries with a record that a read of
    /// `node` at position `at` stands for: those for that frame, and those
    /// whose frame's pops reached it, as `(index, record)`.
    pub(crate) fn landed_payloads(
        &self,
        at: u32,
        node: u32,
        generation: u32,
    ) -> impl Iterator<Item = (usize, u64)> + '_ {
        self.pending
            .standing_for((at, node, generation))
            .into_iter()
            .filter_map(move |i| {
                let d = self.pending.get(i);
                let p = d.provenance.as_ref()?;
                d.is_untaken_panic().then(|| (i, p.record()))
            })
    }

    /// The landing, Dirty or Failed, for the frame of `node` at `cidx`
    /// along `path` whose compute unwound with `payload` after its last
    /// read `last`, as `(reading, ordinal)`, where the landing checks chose
    /// `landing`, if the latest take of the delivery for that read on that
    /// path raised `payload`: its first take records `landing` in
    /// [`Delivery::landed`], and a later one lands Dirty when the first
    /// did. A Dirty `landing` stays: it says a change was in flight, which
    /// a Failed landing would hide. A read after the take means the compute
    /// caught that copy, so a payload at its freed address with the same
    /// text is another panic. The path and seq tell this frame's delivery
    /// from another run's at the position, whose taken copy's freed box can
    /// hold a fresh copy of this one.
    pub(crate) fn retaken_landing(
        &mut self,
        (node, generation, cidx, path, seq): (u32, u32, u32, u64, u32),
        last: (u64, u32),
        payload: &(dyn Any + Send),
        landing: NodeState,
    ) -> Option<NodeState> {
        let unwind = Some(payload_key(payload));
        let i = self
            .pending
            .of_frame(node, generation, cidx)
            .into_iter()
            .find(|&i| {
                let d = self.pending.get(i);
                d.taken
                    && d.key.read_by(node, generation, cidx, last)
                    && d.key.path == path
                    && d.key.seq == seq
                    && d.resumed == unwind
                    && matches!(d.what, Delivered::Unwind { cycle: false, .. })
            })?;
        let d = self.pending.get_mut(i);
        let first = d.landed.get_or_insert(landing);
        Some(if *first == NodeState::Dirty {
            NodeState::Dirty
        } else {
            landing
        })
    }

    /// Takes the payload of delivery `i` for a read at `at`, as
    /// [`Self::take_unwind`] does. When the read is the running frame's
    /// own compute, its unwind carries the record on to
    /// [`Self::unwind_target`].
    pub(crate) fn take_landed(&mut self, i: usize, at: u32) -> (bool, Box<dyn Any + Send>) {
        if let Some(below) = self.unwind_target(at) {
            let key = ReadKey::blocked(below);
            self.provenance = self.pending.get(i).provenance.clone().map(|p| (key, p));
        }
        match self.take_unwind(i) {
            Delivered::Unwind {
                attributed,
                payload,
                ..
            } => (attributed, payload),
            _ => unreachable!("incr-core: matched as an unwind"),
        }
    }

    /// Marks delivery `i`'s read taken, as `take_delivery` does, and
    /// returns its unwind. The delivery stays until its maker's run ends,
    /// for a retry doomed again; a payload that can't be copied is handed
    /// over once, its delivery keeping a [`Regen`] marker in its place.
    pub(crate) fn take_unwind(&mut self, i: usize) -> Delivered {
        let key = self.pending.get(i).key;
        if let Some(p) = &self.pending.get(i).provenance {
            p.spend();
        }
        self.pending.take_keyed(&key);
        self.directs.take_keyed(&key);
        self.pending.pass(i);
        let Delivered::Unwind {
            cycle,
            poison,
            attributed,
            payload,
            nodes,
        } = &self.pending.get(i).what
        else {
            unreachable!("incr-core: taken as an unwind");
        };
        let Some(copy) = copy_payload(&**payload) else {
            // The one payload goes to this read, and a retry doomed again
            // regenerates it.
            let regen: Box<dyn Any + Send> = Box::new(Regen((**payload).type_id()));
            let d = self.pending.get_mut(i);
            let Delivered::Unwind {
                cycle,
                poison,
                attributed,
                payload,
                nodes,
            } = &mut d.what
            else {
                unreachable!("incr-core: taken as an unwind");
            };
            let payload = std::mem::replace(payload, regen);
            let key = payload_key(&*payload);
            let what = Delivered::Unwind {
                cycle: *cycle,
                poison: *poison,
                attributed: *attributed,
                payload,
                nodes: nodes.clone(),
            };
            self.pending.resume(i, key);
            return what;
        };
        let what = Delivered::Unwind {
            cycle: *cycle,
            poison: *poison,
            attributed: *attributed,
            payload: copy,
            nodes: nodes.clone(),
        };
        if let Delivered::Unwind { payload, .. } = &what {
            self.pending.resume(i, payload_key(&**payload));
        }
        what
    }

    /// Keeps the walk of the read virtual frame `below` was blocked in, from
    /// where a doom stopped it, for the frame's retry, with the `runs` (see
    /// [`run_snapshot`]) ended in it so far.
    pub(crate) fn deliver_walk(
        &mut self,
        below: &VFrame,
        walk: crate::runtime::Walk,
        pushed: Option<u32>,
        runs: RunSlot,
    ) {
        let key = ReadKey::blocked(below);
        self.pending.remove_walks(&key);
        let mut d = Delivery::new(key, below.maker, Delivered::Walk(Box::new(walk), pushed));
        d.runs = runs;
        self.pending.push(d);
    }

    /// Sends the read virtual frame `below` was blocked in to its compute
    /// in the retry of the entry that made the frame. A pass at a higher
    /// level can pop the frame above it, and that pass's end must not drop
    /// the delivery before the frame's retry reads it. A kept one keeps its
    /// `runs`: the retry that took it ran none of the read's walk.
    pub(crate) fn deliver_direct(&mut self, below: &VFrame, runs: RunSlot) {
        self.directs
            .deliver(ReadKey::blocked(below), below.maker, runs);
    }

    /// As [`Self::deliver_direct`], for a frame whose compute ran in the
    /// retry that took an earlier walk or direct delivery for the read:
    /// the walk goes and the direct one is read again with `runs`, since
    /// the runs it kept no longer cover the ended ones. A direct one from
    /// the same maker is updated in place, sparing a removal and a push.
    pub(crate) fn redeliver_direct(&mut self, below: &VFrame, runs: RunSlot) {
        let key = ReadKey::blocked(below);
        self.pending.remove_walks(&key);
        self.directs.redeliver(key, below.maker, runs);
    }

    /// The run of the frame at `frame` along `path` is over, and a later
    /// run of its node there is another run: the claim watermark or failure
    /// count kept for the ended run's retry would pass the ended run's
    /// claim or marks on to it.
    pub(crate) fn end_frame_runs(&mut self, frame: (u32, u32, u32), path: u64) {
        self.claim_revs.end_frame(frame, path);
        self.run_failures.end_frame(frame, path);
    }

    /// As [`Self::end_frame_runs`], for every frame of `node` at
    /// `generation`, whose runs a release ended.
    pub(crate) fn end_node_runs(&mut self, node: u32, generation: u32) {
        self.claim_revs.end_node(node, generation);
        self.run_failures.end_node(node, generation);
    }

    /// The run of the entry at `level` ended: the deliveries it made for
    /// its frames' retries are returned to be dropped, and its claim
    /// watermarks and failure counts go.
    pub(crate) fn end_maker(&mut self, level: u32) -> Vec<Delivery> {
        self.claim_revs.end_maker(level);
        self.run_failures.end_maker(level);
        self.directs.remove_maker(level);
        self.pending.extract_maker(level)
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

    /// Truncates `vstack` to `len` with no Tarjan pops, returning the
    /// episodes suspended at the frames removed: they are abandoned, and
    /// their members are released with the claims held there.
    pub(crate) fn truncate_vstack(&mut self, len: usize) -> Vec<Episode> {
        let mut abandoned = Vec::new();
        while self.vstack.len() > len {
            let Some(v) = self.pop_vframe() else {
                break;
            };
            self.end_frame_runs((v.node, v.generation, v.cidx), v.path);
            if let Some(ep) = v.ep {
                if let Some(at) = self.episodes.iter().position(|e| e.id == ep) {
                    abandoned.push(self.episodes.remove(at));
                }
            }
        }
        abandoned
    }

    /// The virtual frames that stand for frames on the unbounded run's
    /// stack: those below the running entry's level. Before the loop
    /// starts, and from that level up, they stand for doomed frames that
    /// have popped, though a read may still meet their positions.
    pub(crate) fn stacked(&self) -> &[VFrame] {
        let level = self.running.map_or(0, |l| l as usize);
        &self.vstack[..level.min(self.vstack.len())]
    }

    /// The highest `vstack` position of `node` at `generation`.
    pub(crate) fn position(&self, node: u32, generation: u32) -> Option<u32> {
        let stack = self.vindex.get(&node)?;
        let &(pos, g) = stack.last()?;
        (g == generation).then_some(pos)
    }
}

/// A read of a frame that may be retried whose outcome the retry's read
/// takes again: the value a landing had left Dirty, or the unwind out of
/// the compute the read ran.
pub(crate) struct KeptRead {
    pub(crate) rt: RuntimeId,
    pub(crate) cidx: u32,
    pub(crate) ctl: *const crate::cycle::ComputeCtl,
    pub(crate) reading: u64,
    pub(crate) ordinal: u32,
    pub(crate) foreign: Option<RuntimeId>,
    pub(crate) what: Delivered,
}

/// The landings copies may hold, as `(runtime, (slot, generation))`.
type CopiedSet = HashSet<(RuntimeId, (u32, u32)), FxBuildHasher>;

#[cfg(not(loom))]
thread_local! {
    static KEPT_READS: RefCell<Vec<(KeptRead, usize)>> = const { RefCell::new(Vec::new()) };
    static CHECKED_READS: RefCell<Vec<CheckedRead>> = const { RefCell::new(Vec::new()) };
    static KEPT_EVENTS: RefCell<Events> = RefCell::new(Events::default());
    static KEPT_TOP: Cell<u32> = const { Cell::new(0) };
    static REGEN: RefCell<Regenerating> = const { RefCell::new(Regenerating::new()) };
    static COPIED: RefCell<CopiedSet> = RefCell::new(CopiedSet::default());
    static COPIED_GEN: Cell<u64> = const { Cell::new(0) };
    static KEPT_HELD: RefCell<CopiedSet> = RefCell::new(CopiedSet::default());
    static KEPT_GEN: Cell<u64> = const { Cell::new(0) };
    // A kept copy holds landings `KEPT_HELD` does not mark: every landing
    // counts as one a kept copy may hold.
    static KEPT_ALL: Cell<bool> = const { Cell::new(false) };
    static CONTEXTS: RefCell<Vec<Context>> = const { RefCell::new(Vec::new()) };
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
    static THREAD_TAG: Cell<u64> = const { Cell::new(0) };
    static SPECULATIVE: Cell<u32> = const { Cell::new(0) };
    static SPECULATING_IN: RefCell<Vec<RuntimeId>> = const { RefCell::new(Vec::new()) };
    static CARRY: Cell<Option<Carry>> = const { Cell::new(None) };
    static MEMBER_RUNS: RefCell<Vec<(u32, (u64, u32))>> = const { RefCell::new(Vec::new()) };
    static SUSPENDED: RefCell<HashMap<u64, (u64, u32), FxBuildHasher>> = RefCell::new(HashMap::default());
    // In step with KEPT_READS: every push and removal there goes through
    // `count_foreign`.
    static FOREIGN_KEPT: RefCell<HashMap<ForeignKey, u32, FxBuildHasher>> = RefCell::new(HashMap::default());
    static RUNS: RefCell<Runs> = RefCell::new(Runs::default());
    static TOOK: RefCell<Takes> = RefCell::new(HashMap::default());
}

#[cfg(loom)]
loom::thread_local! {
    static KEPT_READS: RefCell<Vec<(KeptRead, usize)>> = RefCell::new(Vec::new());
    static CHECKED_READS: RefCell<Vec<CheckedRead>> = RefCell::new(Vec::new());
    static KEPT_EVENTS: RefCell<Events> = RefCell::new(Events::default());
    static KEPT_TOP: Cell<u32> = Cell::new(0);
    static REGEN: RefCell<Regenerating> = RefCell::new(Regenerating::new());
    static COPIED: RefCell<CopiedSet> = RefCell::new(CopiedSet::default());
    static COPIED_GEN: Cell<u64> = Cell::new(0);
    static KEPT_HELD: RefCell<CopiedSet> = RefCell::new(CopiedSet::default());
    static KEPT_GEN: Cell<u64> = Cell::new(0);
    static KEPT_ALL: Cell<bool> = Cell::new(false);
    static CONTEXTS: RefCell<Vec<Context>> = RefCell::new(Vec::new());
    static NEXT_ID: Cell<u64> = Cell::new(1);
    static THREAD_TAG: Cell<u64> = Cell::new(0);
    static SPECULATIVE: Cell<u32> = Cell::new(0);
    static SPECULATING_IN: RefCell<Vec<RuntimeId>> = RefCell::new(Vec::new());
    static CARRY: Cell<Option<Carry>> = Cell::new(None);
    static MEMBER_RUNS: RefCell<Vec<(u32, (u64, u32))>> = RefCell::new(Vec::new());
    static SUSPENDED: RefCell<HashMap<u64, (u64, u32), FxBuildHasher>> = RefCell::new(HashMap::default());
    // In step with KEPT_READS: every push and removal there goes through
    // `count_foreign`.
    static FOREIGN_KEPT: RefCell<HashMap<ForeignKey, u32, FxBuildHasher>> = RefCell::new(HashMap::default());
    static RUNS: RefCell<Runs> = RefCell::new(Runs::default());
    static TOOK: RefCell<Takes> = RefCell::new(HashMap::default());
}

#[cfg(all(test, not(loom)))]
thread_local! {
    static VINDEX_REPUSHES: Cell<u64> = const { Cell::new(0) };
    static NOTE_WORK: Cell<u64> = const { Cell::new(0) };
    static PEAK_LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK_DIRECTS: Cell<usize> = const { Cell::new(0) };
    static PEAK_VSTACK: Cell<usize> = const { Cell::new(0) };
    static PATH_STEPS: Cell<u64> = const { Cell::new(0) };
}

/// Counts `units` elements a pass over this thread's notes, their copies,
/// its kept reads, a context's traced deliveries, the provenance records
/// or a delivery index's list looked at.
#[inline(always)]
pub(crate) fn note_work(units: usize) {
    #[cfg(all(test, not(loom)))]
    NOTE_WORK.with(|n| n.set(n.get() + units as u64));
    #[cfg(not(all(test, not(loom))))]
    let _ = units;
}

/// Runs `f` without counting the note work it does.
#[cfg(all(test, not(loom)))]
fn without_note_work(f: impl FnOnce()) {
    let counted = NOTE_WORK.with(Cell::get);
    f();
    NOTE_WORK.with(|n| n.set(counted));
}

/// The units [`note_work`] counted on this thread since the last call.
#[cfg(all(test, not(loom)))]
pub(crate) fn take_note_work() -> u64 {
    NOTE_WORK.with(|n| n.replace(0))
}

/// The most deliveries, direct ones and virtual frames one context on
/// this thread held at once since the last call.
#[cfg(all(test, not(loom)))]
pub(crate) fn take_peak_live() -> (usize, usize, usize) {
    (
        PEAK_LIVE.with(|p| p.replace(0)),
        PEAK_DIRECTS.with(|p| p.replace(0)),
        PEAK_VSTACK.with(|p| p.replace(0)),
    )
}

/// Counts `frames` stack frames a frame's path was folded over.
#[inline(always)]
pub(crate) fn path_steps(frames: usize) {
    #[cfg(all(test, not(loom)))]
    PATH_STEPS.with(|n| n.set(n.get() + frames as u64));
    #[cfg(not(all(test, not(loom))))]
    let _ = frames;
}

/// The frames [`path_steps`] counted on this thread since the last call.
#[cfg(all(test, not(loom)))]
pub(crate) fn take_path_steps() -> u64 {
    PATH_STEPS.with(|n| n.replace(0))
}

/// Virtual frames pushed on this thread for a node that already had one
/// on its context's `vindex`, since the last call.
#[cfg(all(test, not(loom)))]
pub(crate) fn take_vindex_repushes() -> u64 {
    VINDEX_REPUSHES.with(|n| n.replace(0))
}

/// Whether no context on this thread has a virtual frame indexed.
#[cfg(all(test, not(loom)))]
pub(crate) fn vindexes_empty() -> bool {
    CONTEXTS.with(|c| c.borrow().iter().all(|x| x.vindex.is_empty()))
}

/// An event [`retire_copies`] applied to the kept reads' copies, which
/// take it when a retry takes them.
struct Logged {
    rt: RuntimeId,
    ev: Retire,
    /// The landing's node may be in any copy, so the event reaches copies
    /// of frames at or above it too.
    all: bool,
    /// The event may retire landings by position, short of the node.
    wide: bool,
    /// The landing's node may be in a kept read's copy.
    kept: bool,
}

/// The events logged while reads are kept, with the indexes of those that
/// may retire landings by position and of the landings of nodes a kept
/// copy held, by node and in log order, so that a kept read's catch-up
/// skips the landings of nodes its copy does not hold, and with none since
/// it was kept, looks at no node of its copy.
#[derive(Default)]
struct Events {
    log: Vec<Logged>,
    wide: Vec<usize>,
    landed: HashMap<(RuntimeId, (u32, u32)), Vec<usize>, FxBuildHasher>,
    kept: Vec<usize>,
}

impl Events {
    fn push(&mut self, l: Logged) {
        let i = self.log.len();
        if l.wide {
            self.wide.push(i);
        }
        if let Some(n) = l.ev.node().filter(|_| l.kept) {
            self.landed.entry((l.rt, n)).or_default().push(i);
            self.kept.push(i);
        }
        self.log.push(l);
    }

    fn clear(&mut self) {
        self.log.clear();
        self.wide.clear();
        self.kept.clear();
        // As for the copied set, a clear costs the capacity.
        if self.landed.capacity() > COPIED_KEEP {
            self.landed = HashMap::default();
        } else {
            self.landed.clear();
        }
    }
}

/// Keeps `read` for its frame's retry. Reads of frames that have popped go:
/// every frame at or above the reading one's position, but the reading
/// one, which is a run of its own if this is its first such read.
pub(crate) fn keep_read(read: KeptRead, fresh: bool) {
    keep(read, fresh, false);
}

/// [`keep_read`] for a read of another runtime's node whose ordinal (see
/// [`foreign_ordinal`]) no take needed first: the count that keeping it
/// raises gives it.
pub(crate) fn keep_foreign_read(read: KeptRead, fresh: bool) {
    keep(read, fresh, true);
}

/// A speculative frame's Clean read of another runtime, with that
/// runtime's count of changes as the read found it, which the frame's
/// stash is adopted only while it still holds.
struct CheckedRead {
    rt: RuntimeId,
    cidx: u32,
    ctl: *const crate::cycle::ComputeCtl,
    seen: Count,
}

/// Keeps `seen` for the speculative frame of `ctl` at `cidx` of `rt`, as
/// [`keep_read`] keeps the read: the checked reads of frames at or above its
/// position, which have popped, go, and so do the frame's own ones on its
/// first kept read (`fresh`), which an earlier frame there left.
pub(crate) fn check_read(
    rt: RuntimeId,
    cidx: u32,
    ctl: *const crate::cycle::ComputeCtl,
    seen: Count,
    fresh: bool,
) {
    CHECKED_READS.with(|d| {
        let mut d = d.borrow_mut();
        d.retain(|e| e.rt != rt || e.cidx < cidx || !fresh && e.cidx == cidx && e.ctl == ctl);
        let ours = |e: &CheckedRead| e.rt == rt && e.cidx == cidx && e.ctl == ctl;
        if !d
            .iter_mut()
            .rev()
            .take_while(|e| ours(e))
            .any(|e| e.seen.absorb(&seen))
        {
            d.push(CheckedRead {
                rt,
                cidx,
                ctl,
                seen,
            });
        }
    });
}

/// A context named so that no context of another thread, now or later, has
/// the same name: context ids count per thread.
pub(crate) type ContextName = (u64, u64);

fn thread_tag() -> u64 {
    static NEXT_THREAD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    THREAD_TAG.with(|t| {
        if t.get() == 0 {
            t.set(NEXT_THREAD.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        }
        t.get()
    })
}

/// This thread's outermost context, of any runtime: contexts nest, so some
/// context open now is still open later exactly when this one is.
pub(crate) fn outermost_context() -> Option<ContextName> {
    let thread = thread_tag();
    peek(|c| c.first().map(|x| (thread, x.id)))
}

/// Whether context `name` is open on this thread, at any depth.
pub(crate) fn context_open(name: ContextName) -> bool {
    name.0 == thread_tag() && peek(|c| c.iter().any(|x| x.id == name.1))
}

/// The checked reads of the frame of `ctl` at `cidx` of `rt`, which is
/// popping, taken with those of frames above it.
pub(crate) fn take_checked_reads(
    rt: RuntimeId,
    cidx: u32,
    ctl: *const crate::cycle::ComputeCtl,
) -> Vec<Count> {
    CHECKED_READS.with(|d| {
        let mut counts = Vec::new();
        for e in d
            .borrow_mut()
            .extract_if(.., |e| e.rt == rt && e.cidx >= cidx)
            .filter(|e| e.cidx == cidx && e.ctl == ctl)
        {
            crate::changes::fold(&mut counts, &e.seen);
        }
        counts
    })
}

fn keep(mut read: KeptRead, fresh: bool, number: bool) {
    if let Delivered::Unwind { nodes, .. } = &read.what {
        let top = read.cidx + 1 + nodes.depth();
        KEPT_TOP.with(|k| k.set(k.get().max(top)));
        // A shared copy marks only the notes no earlier copy of its log
        // did; marking one taken apart would cost its size at every keep,
        // so every landing counts as kept instead.
        if nodes.as_shared().is_some() {
            mark_in(Marks::Kept, read.rt, nodes);
        } else if !matches!(&nodes.nodes, Nodes::Owned(n) if n.is_empty()) {
            KEPT_ALL.with(|k| k.set(true));
        }
    }
    let seen = KEPT_EVENTS.with(|v| v.borrow().log.len());
    let gone: Vec<(KeptRead, usize)> = KEPT_READS.with(|d| {
        let mut d = d.borrow_mut();
        let ours = |e: &KeptRead| e.rt == read.rt && e.cidx == read.cidx && e.ctl == read.ctl;
        // The frame's latest kept read went in under this same rule, so
        // nothing before it goes now.
        let from = d
            .iter()
            .rposition(|(e, _)| e.cidx < read.cidx || !fresh && ours(e))
            .map_or(0, |i| i + 1);
        note_work(d.len() - from);
        let goes = |e: &KeptRead| e.rt == read.rt && e.cidx >= read.cidx && (fresh || !ours(e));
        // Mostly the frame's own latest read is all there is past `from`.
        let gone: Vec<(KeptRead, usize)> = if d[from..].iter().any(|(e, _)| goes(e)) {
            d.extract_if(from.., |(e, _)| goes(e)).collect()
        } else {
            Vec::new()
        };
        uncount_foreign(&gone, d.is_empty());
        // Most replay coverage runs in release, and the scans of kept reads
        // rely on this.
        if cfg!(any(test, debug_assertions)) {
            assert!(
                d.last().is_none_or(|(e, _)| e.cidx <= read.cidx),
                "incr-core: kept reads run in frame order"
            );
        }
        let count = count_foreign(&read, true);
        if number {
            read.ordinal = count;
        } else {
            debug_assert!(
                read.foreign.is_none() || read.ordinal == count,
                "incr-core: a kept foreign read's ordinal is not its count"
            );
        }
        d.push((read, seen));
        gone
    });
    value_code::set_kept_bit(true);
    let _drops = crate::attempt::Outside::enter(true);
    drop(gone);
}

/// The index of the first of `kept` that a frame at `cidx` or above
/// made. Kept reads are pushed by the stack's top frame after those at or
/// above it go, and go when their frame pops, so they run in frame order.
fn from_frame(kept: &[(KeptRead, usize)], cidx: u32) -> usize {
    let from = kept
        .iter()
        .rposition(|(e, _)| e.cidx < cidx)
        .map_or(0, |i| i + 1);
    note_work(kept.len() - from);
    from
}

/// The capacity of the copied set kept across a clear.
const COPIED_KEEP: usize = 1024;

/// Marks `nodes` as landings a copy may hold, so that a later landing of
/// one of them finds the contexts' deliveries to drop it from. Only a
/// context's deliveries need the marks: a kept read applies a landing to
/// its copy when the copy holds the node (`catch_up`), and a context that
/// opens marks the kept reads' copies.
pub(crate) fn mark_copied(rt: RuntimeId, nodes: &Landings) {
    if value_code::thread_word() & value_code::REPLAY_BIT != 0 {
        mark(rt, nodes);
    }
}

fn mark(rt: RuntimeId, nodes: &Landings) {
    mark_in(Marks::Copied, rt, nodes);
}

/// Which set of landings copies may hold: the contexts' deliveries'
/// (`COPIED`) or the kept reads' (`KEPT_HELD`).
#[derive(Clone, Copy)]
enum Marks {
    Copied,
    Kept,
}

fn log_marks(log: &mut Log, marks: Marks) -> &mut Marked {
    match marks {
        Marks::Copied => &mut log.marked,
        Marks::Kept => &mut log.kept_marked,
    }
}

/// A shared copy marks only the notes its log's earlier copies did not.
fn mark_in(marks: Marks, rt: RuntimeId, nodes: &Landings) {
    let (generation, set) = match marks {
        Marks::Copied => (copied_generation(), &COPIED),
        Marks::Kept => (kept_generation(), &KEPT_HELD),
    };
    set.with(|c| {
        let mut c = c.borrow_mut();
        let mut mark = |n| {
            c.insert((rt, n));
        };
        match &nodes.nodes {
            Nodes::Owned(nodes) => {
                note_work(nodes.len());
                for &(n, _, _) in nodes {
                    mark(n);
                }
            }
            // A cut copy holds less than its log up to `len`, so it marks
            // only what it holds and leaves the watermark to uncut copies.
            Nodes::Shared {
                log,
                len,
                epoch,
                base,
                cut: Some(cut),
                ..
            } => {
                for (n, _, _) in log.borrow().landings(*len, *epoch, *base, Some(cut)) {
                    mark(n);
                }
            }
            Nodes::Shared {
                log, len, epoch, ..
            } => {
                let mut log = log.borrow_mut();
                let from = log_marks(&mut log, marks).advance(generation, *len, *epoch);
                note_work(len - from);
                for e in log.entries[from..*len].iter().filter(|e| e.gone >= *epoch) {
                    match &e.lent {
                        None => mark(e.node),
                        // A ticket's members outlive the kept set, so
                        // marks on them would restart with every set: the
                        // copy counts every landing instead.
                        Some(_) if matches!(marks, Marks::Kept) => {
                            KEPT_ALL.with(|k| k.set(true));
                        }
                        // A ticket's members went into earlier copies of
                        // the notes that stood for it, so only its later
                        // members are new.
                        Some((members, len)) => {
                            let mut joined = members.lock();
                            let from = joined.marked.advance(generation, *len, *epoch);
                            note_work(len - from);
                            for m in joined.members[from..*len]
                                .iter()
                                .filter(|m| m.gone >= *epoch)
                            {
                                mark(m.node);
                            }
                        }
                    }
                }
            }
        }
    });
}

/// This thread's copied set's generation, unique among every thread's, as
/// a ticket's members are marked from any thread.
fn copied_generation() -> u64 {
    COPIED_GEN.with(|g| {
        if g.get() == 0 {
            g.set(new_generation());
        }
        g.get()
    })
}

/// As [`copied_generation`], for the kept reads' set.
fn kept_generation() -> u64 {
    KEPT_GEN.with(|g| {
        if g.get() == 0 {
            g.set(new_generation());
        }
        g.get()
    })
}

fn new_generation() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Whether a kept read or a context's delivery may hold copied landings.
pub(crate) fn copies_held() -> bool {
    value_code::thread_word() & (value_code::KEPT_BIT | value_code::REPLAY_BIT) != 0
}

/// Applies `ev`, an event that retires notes of runtime `rt`, to the
/// copies of them kept reads and the contexts' passed deliveries hold,
/// outside a speculative run. A kept read takes the events when a retry
/// takes it, so an event costs nothing per kept read.
pub(crate) fn retire_copies(rt: RuntimeId, ev: &Retire) {
    let all = ev
        .node()
        .is_some_and(|n| COPIED.with(|c| c.borrow().contains(&(rt, n))));
    // Short of a copied node, an event retires only landings above its
    // position, or at it for a contained payload; a landing also retires
    // its node from the kept copies that hold it, which the kept set
    // marks and the copied set does not.
    let reach = match *ev {
        Retire::Contained { cidx, .. } => cidx,
        _ => ev.cidx() + 1,
    };
    if value_code::thread_word() & value_code::KEPT_BIT != 0 {
        let wide = all || KEPT_TOP.with(Cell::get) >= reach;
        let kept = ev.node().is_some_and(|n| {
            KEPT_ALL.with(Cell::get) || KEPT_HELD.with(|k| k.borrow().contains(&(rt, n)))
        });
        if wide || kept {
            KEPT_EVENTS.with(|v| {
                v.borrow_mut().push(Logged {
                    rt,
                    ev: *ev,
                    all,
                    wide,
                    kept,
                })
            });
        }
    }
    with(|c| {
        for x in c.iter_mut().filter(|x| x.rt == rt) {
            x.pending.retire(ev, all);
        }
    });
}

/// Applies to `read`'s copy the events logged since it was kept, as each
/// applied when it happened: to a read by a frame below the event, of any
/// frame for a copied node's landing, or holding the landing's node then.
/// An event that is not wide changes the copy only through its node, so
/// only the landings of nodes the copy holds now are looked at.
fn catch_up(read: &mut KeptRead, seen: usize, events: &Events) {
    let Delivered::Unwind { nodes, .. } = &mut read.what else {
        return;
    };
    let (rt, cidx) = (read.rt, read.cidx);
    let from = events.wide.partition_point(|&i| i < seen);
    // `(index, whether the copy held its node)`
    let mut todo: Vec<(usize, bool)> = events.wide[from..].iter().map(|&i| (i, false)).collect();
    if events.kept.last().is_some_and(|&i| i >= seen) {
        let copy = nodes.owned_nodes();
        note_work(copy.len());
        for &(n, _, _) in copy.iter() {
            if let Some(at) = events.landed.get(&(rt, n)) {
                todo.extend(
                    at[at.partition_point(|&i| i < seen)..]
                        .iter()
                        .map(|&i| (i, true)),
                );
            }
        }
        todo.sort_unstable();
        todo.dedup_by(|cur, prev| {
            let same = cur.0 == prev.0;
            prev.1 |= same && cur.1;
            same
        });
    }
    note_work(todo.len());
    for (i, held) in todo {
        let l = &events.log[i];
        if l.rt == rt
            && (l.wide && (cidx < l.ev.cidx() || l.all)
                || held && l.ev.node().is_some_and(|n| nodes.holds(n)))
        {
            nodes.retire(cidx, &l.ev, None);
        }
    }
}

/// Catches up the kept reads of `rt`'s frame at `cidx`, each `(read,
/// seen)`, on the events logged since it was kept. Every event only takes
/// landings away, and the events a copy missed take the same in any order:
/// the landings deeper than the shallowest position one of them takes, and
/// the nodes landed since. So a copy that shares its log keeps sharing it,
/// with those cut ([`Cut`]), and one backward pass over the events gives
/// every copy its cut. A copy taken apart, or one a held landing below the
/// frame may empty, takes the events one at a time ([`catch_up`]).
fn catch_up_frame(rt: RuntimeId, cidx: u32, reads: &mut [(KeptRead, usize)], events: &Events) {
    let shared = |e: &KeptRead| {
        matches!(&e.what, Delivered::Unwind { nodes, .. }
            if matches!(&nodes.nodes, Nodes::Shared { cut: None, .. }))
    };
    let Some(first) = reads
        .iter()
        .filter(|(e, _)| shared(e))
        .map(|(_, s)| *s)
        .min()
    else {
        for (e, seen) in reads.iter_mut() {
            catch_up(e, *seen, events);
        }
        return;
    };
    let (landed, below) = landed_since(rt, cidx, first, events);
    let landed = Rc::new(landed);
    let mut order: Vec<usize> = (0..reads.len()).collect();
    order.sort_unstable_by_key(|&k| std::cmp::Reverse(reads[k].1));
    let wide = &events.wide[events.wide.partition_point(|&i| i < first)..];
    note_work(wide.len());
    let mut cuts = CutPositions::default();
    let mut next = wide.len();
    for k in order {
        let (e, seen) = &mut reads[k];
        if !shared(e) || below.is_some_and(|b| b >= *seen) {
            catch_up(e, *seen, events);
            continue;
        }
        while next > 0 && wide[next - 1] >= *seen {
            next -= 1;
            let l = &events.log[wide[next]];
            if l.rt == rt && (cidx < l.ev.cidx() || l.all) {
                cuts.add(&l.ev);
            }
        }
        let Delivered::Unwind { nodes, .. } = &mut e.what else {
            unreachable!()
        };
        let at = cuts.position(nodes.source);
        if at <= i64::from(cidx) {
            nodes.nodes = Nodes::Owned(Vec::new());
            continue;
        }
        let Nodes::Shared { depth, cut, .. } = &mut nodes.nodes else {
            unreachable!()
        };
        let keep = u32::try_from(at - i64::from(cidx) - 1).unwrap_or(u32::MAX);
        if keep < *depth || landed.order.last().is_some_and(|&i| i >= *seen) {
            *depth = (*depth).min(keep);
            *cut = Some(Cut {
                depth: keep,
                seen: *seen,
                landed: Rc::clone(&landed),
            });
        }
    }
}

/// The landings of `rt`'s nodes kept copies held, logged from index
/// `first` on, and the latest of them at or below the frame at `cidx`: a
/// copy holding that node loses every landing of another payload.
fn landed_since(
    rt: RuntimeId,
    cidx: u32,
    first: usize,
    events: &Events,
) -> (CutLandings, Option<usize>) {
    let mut landed = CutLandings::default();
    let mut below = None;
    let kept = &events.kept[events.kept.partition_point(|&i| i < first)..];
    note_work(kept.len());
    for &i in kept {
        let l = &events.log[i];
        if l.rt != rt {
            continue;
        }
        if let Retire::Landing { cidx: at, node, .. } = l.ev {
            landed.latest.insert(node, i);
            landed.order.push(i);
            if at <= cidx {
                below = Some(i);
            }
        }
    }
    (landed, below)
}

/// The positions the events a backward pass has met take landings above,
/// for a copy of the notes of a payload: an event takes those above its
/// position, a containment of the copy's own payload also those at it, and
/// a landing of another payload's those above it.
#[derive(Default)]
struct CutPositions {
    any: Option<i64>,
    /// Per payload, what its events take from a copy of another payload's
    /// notes and from a copy of its own.
    keyed: HashMap<PayloadKey, (i64, i64), FxBuildHasher>,
    /// The two payloads with the lowest positions for others' copies.
    best: [Option<(i64, PayloadKey)>; 2],
}

impl CutPositions {
    fn add(&mut self, ev: &Retire) {
        match *ev {
            Retire::Above(at) => self.any = Some(self.any.map_or(at.into(), |a| a.min(at.into()))),
            Retire::Contained { cidx: at, payload } => {
                self.keyed_add(payload, i64::from(at), i64::from(at) - 1);
            }
            Retire::Landing {
                cidx: at,
                payload: Some(payload),
                ..
            } => self.keyed_add(payload, i64::from(at), i64::MAX),
            Retire::Landing { payload: None, .. } => {}
        }
    }

    fn keyed_add(&mut self, key: PayloadKey, others: i64, own: i64) {
        let e = self.keyed.entry(key).or_insert((i64::MAX, i64::MAX));
        e.0 = e.0.min(others);
        e.1 = e.1.min(own);
        let v = e.0;
        if let Some(i) = self
            .best
            .iter()
            .position(|b| b.is_some_and(|(_, k)| k == key))
        {
            self.best[i] = Some((v, key));
        } else if self.best[1].is_none_or(|(b, _)| v < b) {
            self.best[1] = Some((v, key));
        }
        if self.best[1].is_some_and(|(b1, _)| self.best[0].is_none_or(|(b0, _)| b1 < b0)) {
            self.best.swap(0, 1);
        }
    }

    /// The lowest position the events take landings above in a copy of
    /// `source`'s notes; `i64::MAX` if none.
    fn position(&self, source: Option<PayloadKey>) -> i64 {
        let own = source
            .and_then(|s| self.keyed.get(&s))
            .map_or(i64::MAX, |e| e.1);
        let others = self
            .best
            .iter()
            .flatten()
            .find(|(_, k)| Some(*k) != source)
            .map_or(i64::MAX, |(v, _)| *v);
        self.any.unwrap_or(i64::MAX).min(own).min(others)
    }
}

/// Marks the kept reads' copies for a context that opens.
fn mark_kept() {
    KEPT_READS.with(|d| {
        for (e, _) in d.borrow().iter() {
            if let Delivered::Unwind { nodes, .. } = &e.what {
                mark(e.rt, nodes);
            }
        }
    });
}

/// The reads kept for the run of control block `ctl` at `cidx`.
pub(crate) fn take_kept_reads(
    rt: RuntimeId,
    cidx: u32,
    ctl: *const crate::cycle::ComputeCtl,
) -> Vec<KeptRead> {
    let mut out = extract_kept(cidx, |e| e.rt == rt && e.cidx == cidx && e.ctl == ctl);
    KEPT_EVENTS.with(|v| catch_up_frame(rt, cidx, &mut out, &v.borrow()));
    if value_code::thread_word() & value_code::KEPT_BIT == 0 {
        forget_events();
    }
    out.into_iter().map(|(e, _)| e).collect()
}

/// Drops the reads kept for frames of `rt` at or above `cidx`, which a
/// frame pop at `cidx` ends: only a doom takes them, and it took them as
/// it marked the frames.
#[cold]
#[inline(never)]
pub(crate) fn drop_kept_from(rt: RuntimeId, cidx: u32) {
    let gone = extract_kept(cidx, |e| e.rt == rt && e.cidx >= cidx);
    if value_code::thread_word() & value_code::KEPT_BIT == 0 {
        forget_events();
    }
    let _drops = crate::attempt::Outside::enter(true);
    drop(gone);
}

/// Taken out of the list before they drop, since a kept value's `Drop`
/// runs user code. `which` takes only reads of frames at `cidx` or above.
fn extract_kept(cidx: u32, mut which: impl FnMut(&KeptRead) -> bool) -> Vec<(KeptRead, usize)> {
    KEPT_READS.with(|d| {
        let mut d = d.borrow_mut();
        let from = from_frame(&d, cidx);
        let mut out = Vec::with_capacity(d.len() - from);
        out.extend(d.extract_if(from.., |(e, _)| which(e)));
        uncount_foreign(&out, d.is_empty());
        debug_assert!(
            !d.is_empty() || FOREIGN_KEPT.with(|m| m.borrow().is_empty()),
            "incr-core: a foreign read's count outlived its kept read"
        );
        value_code::set_kept_bit(!d.is_empty());
        let word = value_code::thread_word();
        if d.is_empty() && word & value_code::REPLAY_BIT == 0 {
            COPIED.with(clear_set);
            COPIED_GEN.with(|g| g.set(0));
            if !speculative() {
                CHECKED_READS.with(|c| c.borrow_mut().clear());
            }
            if word & value_code::NOTES_BIT == 0 {
                end_kept_marks();
            }
        }
        out
    })
}

/// Clears the kept set once no read is kept, no context is open and the
/// thread holds no notes. While notes stay, a later read may keep a longer
/// copy of a log whose earlier entries are marked, and clearing then
/// would mark them again for every read up a chain.
pub(crate) fn end_kept_marks() {
    KEPT_HELD.with(clear_set);
    KEPT_GEN.with(|g| g.set(0));
    KEPT_ALL.with(|k| k.set(false));
}

fn clear_set(set: &RefCell<CopiedSet>) {
    let mut set = set.borrow_mut();
    // A clear costs the capacity, which one big replay leaves behind for
    // every later kept episode.
    if set.capacity() > COPIED_KEEP {
        *set = CopiedSet::default();
    } else {
        set.clear();
    }
}

/// No read is kept, so no copy takes the logged events.
fn forget_events() {
    KEPT_EVENTS.with(|v| v.borrow_mut().clear());
    KEPT_TOP.with(|k| k.set(0));
}

/// Which of its reads of `reading`, a node of runtime `foreign`, the read
/// the frame of `ctl` at `cidx` of `rt` makes now is, from 1. `kept` says
/// whether the frame has kept a read yet, so that reads kept for an earlier
/// frame there are not counted.
pub(crate) fn foreign_ordinal(
    rt: RuntimeId,
    cidx: u32,
    ctl: *const crate::cycle::ComputeCtl,
    reading: u64,
    foreign: RuntimeId,
    kept: bool,
) -> u32 {
    if !kept {
        return 1;
    }
    note_work(1);
    let key = (rt, cidx, ctl as usize, reading, foreign);
    let earlier = FOREIGN_KEPT.with(|m| m.borrow().get(&key).copied().unwrap_or(0));
    #[cfg(debug_assertions)]
    KEPT_READS.with(|d| {
        let scanned = d
            .borrow()
            .iter()
            .filter(|(e, _)| foreign_key(e) == Some(key))
            .count();
        assert_eq!(
            scanned, earlier as usize,
            "incr-core: a frame's counted and scanned foreign reads differ"
        );
    });
    earlier + 1
}

/// A kept foreign read's frame (runtime, position and control block), the
/// node it read and that node's runtime.
type ForeignKey = (RuntimeId, u32, usize, u64, RuntimeId);

fn foreign_key(e: &KeptRead) -> Option<ForeignKey> {
    e.foreign
        .map(|foreign| (e.rt, e.cidx, e.ctl as usize, e.reading, foreign))
}

/// Counts out the reads in `gone`, which have left the kept reads; when
/// `none_left`, none is kept, so no count is left either. Debug builds
/// count each out even then, so a count the reads lost track of panics
/// rather than going with the clear.
fn uncount_foreign(gone: &[(KeptRead, usize)], none_left: bool) {
    if gone.is_empty() {
        return;
    }
    if none_left && !cfg!(debug_assertions) {
        FOREIGN_KEPT.with(|m| m.borrow_mut().clear());
        return;
    }
    for (e, _) in gone {
        count_foreign(e, false);
    }
    debug_assert!(
        !none_left || FOREIGN_KEPT.with(|m| m.borrow().is_empty()),
        "incr-core: a foreign read's count outlived its kept read"
    );
}

/// Counts `e` in or out of `FOREIGN_KEPT`, which holds, for each
/// [`ForeignKey`], how many of the kept reads are under it, and returns
/// how many are now (0 for a read of this runtime's).
fn count_foreign(e: &KeptRead, kept: bool) -> u32 {
    let Some(key) = foreign_key(e) else {
        return 0;
    };
    FOREIGN_KEPT.with(|m| {
        let mut m = m.borrow_mut();
        match m.entry(key) {
            MapEntry::Occupied(mut o) if !kept => {
                *o.get_mut() -= 1;
                let left = *o.get();
                if left == 0 {
                    o.remove();
                }
                left
            }
            MapEntry::Occupied(mut o) => {
                *o.get_mut() += 1;
                *o.get()
            }
            MapEntry::Vacant(v) if kept => *v.insert(1),
            MapEntry::Vacant(_) => {
                unreachable!("incr-core: a foreign read left the kept reads uncounted")
            }
        }
    })
}

/// The id the next context or episode gets.
pub(crate) fn peek_id() -> u64 {
    NEXT_ID.with(Cell::get)
}

/// A frame as the paths of the frames above it see it: its run as well as
/// its read, since the frame above a read is not always the read's node (a
/// walk computes the node's deps first), and an episode's member run reads
/// as the discovery run of its node did.
#[derive(Clone, Copy)]
pub(crate) struct Step {
    pub(crate) node: u32,
    /// The episode of the frame's member run, if it is one.
    pub(crate) episode: Option<u64>,
    /// The node the frame reads, `slot | generation << 32`.
    pub(crate) reading: u64,
    /// Which of the frame's reads of that node it is in, from 1.
    pub(crate) ordinal: u32,
    /// The frame's run's seq (see [`Context::runs`]).
    pub(crate) seq: u32,
}

impl Step {
    /// The step of `rt`'s frame of `node` on the stack, a member run if
    /// `member`.
    pub(crate) fn of(rt: RuntimeId, node: u32, member: bool, reading: u64, ordinal: u32) -> Self {
        Self {
            node,
            episode: member.then(|| held_episode(rt, node)).flatten(),
            reading,
            ordinal,
            seq: 0,
        }
    }

    /// The step of the frame `v` stands for.
    pub(crate) fn virtual_frame(v: &VFrame) -> Self {
        Self {
            node: v.node,
            episode: v.member_of,
            reading: v.reading,
            ordinal: v.ordinal,
            seq: v.seq,
        }
    }
}

/// The path of a frame above one at `path` that is in `step`. A first
/// run's `seq` is left out, so its paths are those of a frame with no
/// earlier run.
pub(crate) fn extend_path(path: u64, step: Step) -> u64 {
    [
        u64::from(step.node),
        step.episode.map_or(0, |e| e + 1),
        step.reading,
        u64::from(step.ordinal) + 1,
    ]
    .into_iter()
    .chain((step.seq != 0).then_some(u64::from(step.seq)))
    .fold(path, |p, x| {
        p.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(x)
    })
}

/// A read that may take deliveries: when it ends, the walk directions it
/// took, used or not, go with it, so that no later read of the node finds
/// them.
pub(crate) struct ReadScope {
    pub(crate) rt: RuntimeId,
    pub(crate) reading: u64,
}

impl Drop for ReadScope {
    fn drop(&mut self) {
        with(|c| {
            for x in c.iter_mut().filter(|x| x.rt == self.rt) {
                x.direct = None;
                x.resume.take_if(|r| r.0 == self.reading);
            }
        });
    }
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
        let found = c.iter_mut().rev().find_map(|x| {
            let level = x.vstack.iter().rposition(|v| v.ep == Some(ep.id))?;
            (x.rt == rt).then_some((x, level as u32))
        });
        match found {
            Some((x, level)) => {
                SUSPENDED.with(|s| s.borrow_mut().insert(ep.id, (x.id, level)));
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

/// The context and level of the member run's virtual frame of episode
/// `ep`, if it waits in a context.
fn suspension(c: &[Context], ep: u64) -> Option<(usize, u32)> {
    if ep == 0 {
        return None;
    }
    let (id, level) = SUSPENDED.with(|s| s.borrow().get(&ep).copied())?;
    let at = c.iter().position(|x| x.id == id)?;
    c[at]
        .episodes
        .iter()
        .any(|e| e.id == ep)
        .then_some((at, level))
}

/// Whether episode `ep` waits in a context: its members wait for its
/// member run's virtual frame, whatever runs at that position meanwhile.
pub(crate) fn is_suspended(ep: u64) -> bool {
    ep != 0 && with(|c| suspension(c, ep).is_some())
}

/// For a read of a member of episode `ep`: `None` if the episode is not
/// suspended, else whether the read is made by an entry its context runs
/// above the member run's virtual frame, the only runs the unbounded run
/// has while the member run is blocked. Any other read comes from a
/// doomed frame's run below, which the unbounded run makes only after the
/// episode lands.
pub(crate) fn suspended_reachable(ep: u64) -> Option<bool> {
    if ep == 0 {
        return None;
    }
    with(|c| {
        let (at, level) = suspension(c, ep)?;
        Some(c[at].running.is_some_and(|l| level < l))
    })
}

/// Whether a claim of runtime `rt` anchored at `anchor`, released from
/// position `cidx`, waits for an episode that waits in a context with its
/// root between them: the unbounded run's root still holds the claim, which
/// the episode joins when it resumes.
pub(crate) fn waits_for_suspended(rt: RuntimeId, anchor: u32, cidx: u32) -> bool {
    with(|c| {
        c.iter()
            .filter(|x| x.rt == rt)
            .flat_map(|x| x.episodes.iter())
            .any(|e| cidx <= e.cidx && e.cidx <= anchor)
    })
}

/// Whether an episode of runtime `rt` that waits in a context has a member
/// run that read the frame at `cidx`, below its root: resumed, it defers
/// its members to that frame.
pub(crate) fn suspended_low_at(rt: RuntimeId, cidx: u32) -> bool {
    with(|c| {
        c.iter()
            .filter(|x| x.rt == rt)
            .flat_map(|x| x.episodes.iter())
            .any(|e| e.cidx > cidx && e.round.as_ref().is_some_and(|r| r.low == cidx))
    })
}

/// Takes episode `ep` out of context `id`.
pub(crate) fn take_episode(id: u64, ep: u64) -> Option<Box<Episode>> {
    SUSPENDED.with(|s| s.borrow_mut().remove(&ep));
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

/// The kind an unwind of `kind` out of a run goes on as. A speculative
/// run may evaluate nodes the unbounded run never does, so a panic in it
/// lands nothing Failed, and a cycle it raises is no read's result: the
/// real run raises either again if it does reach the node. Either goes on
/// as a discard.
pub(crate) fn run_unwind_kind(kind: Option<UnwindKind>, speculative: bool) -> Option<UnwindKind> {
    if speculative && matches!(kind, None | Some(UnwindKind::Cycle)) {
        Some(UnwindKind::Discard)
    } else {
        kind
    }
}

/// Whether a doomed run of node `slot` at `generation` of `rt` left a
/// lowlink or a claim watermark for the run that finishes its frame, which
/// an adopted stash would not take.
pub(crate) fn frame_records(rt: RuntimeId, slot: u32, generation: u32) -> bool {
    with(|c| {
        c.iter().filter(|x| x.rt == rt).any(|x| {
            x.lows.iter().any(|l| l.0 == slot && l.1 == generation)
                || x.claim_revs.has_node(slot, generation)
        })
    })
}

/// Counts a speculative run while it lives, with the runtime whose chain
/// speculates; dropping it, by an unwind included, gives both back.
pub(crate) struct Speculating(());

impl Speculating {
    pub(crate) fn enter(chain: RuntimeId) -> Self {
        SPECULATIVE.with(|s| s.set(s.get() + 1));
        SPECULATING_IN.with(|c| c.borrow_mut().push(chain));
        Self(())
    }
}

impl Drop for Speculating {
    fn drop(&mut self) {
        SPECULATING_IN.with(|c| c.borrow_mut().pop());
        SPECULATIVE.with(|s| s.set(s.get() - 1));
    }
}

/// Whether a speculative run on this thread speculates in a chain of `rt`.
pub(crate) fn speculating_in(rt: RuntimeId) -> bool {
    speculative() && SPECULATING_IN.with(|c| c.borrow().contains(&rt))
}

#[cfg(test)]
std::thread_local! {
    static SPECULATIVE_WAIT_DROPS: Cell<u32> = const { Cell::new(0) };
}

/// Counts, on this thread, a speculative run dropped where it would
/// register a claim wait.
#[cfg(test)]
pub(crate) fn count_speculative_wait_drop() {
    SPECULATIVE_WAIT_DROPS.with(|d| d.set(d.get() + 1));
}

/// The speculative wait drops this thread counted since the last take.
#[cfg(all(test, not(loom)))]
pub(crate) fn take_speculative_wait_drops() -> u32 {
    SPECULATIVE_WAIT_DROPS.with(|d| d.replace(0))
}

/// Runs `f` on the thread's contexts without changing them, so the replay
/// bit [`with`] keeps stays as it is. `f` must not call user code or the
/// engine.
pub(crate) fn peek<R>(f: impl FnOnce(&[Context]) -> R) -> R {
    CONTEXTS.with(|c| f(&c.borrow()))
}

/// Whether the thread's contexts are borrowed: value code runs inside
/// [`with`], and so would a user `Drop` of a value the contexts held if
/// an unwind out of a closure dropped it there.
pub(crate) fn held() -> bool {
    CONTEXTS.with(|c| c.try_borrow_mut().is_err())
}

#[cfg(all(test, not(loom)))]
thread_local! {
    static HELD_DROPS: Cell<u32> = const { Cell::new(0) };
}

/// Counts an unbuilt value dropped inside the contexts' borrow, which
/// only an unwind out of a [`with`] closure may do. The contexts' own
/// drop at thread exit, when they can't be reached, counts nothing.
#[cfg(all(test, not(loom)))]
pub(crate) fn note_drop() {
    if CONTEXTS
        .try_with(|c| c.try_borrow_mut().is_err())
        .unwrap_or(false)
    {
        HELD_DROPS.with(|n| n.set(n.get() + 1));
    }
}

#[cfg(all(test, not(loom)))]
pub(crate) fn take_held_drops() -> u32 {
    HELD_DROPS.with(|n| n.replace(0))
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

/// The Failed read a provenance record named, until a later Failed read
/// of the same node on this thread or the end of the outermost entry.
pub(crate) fn carry() -> Option<Carry> {
    CARRY.with(Cell::get)
}

pub(crate) fn set_carry(carry: Option<Carry>) {
    CARRY.with(|c| c.set(carry));
}

/// What a kept read holds in place of a payload no copy can stand for,
/// once the one payload is gone: the payload's type, which the retry's
/// read regenerates by running the node's closure again
/// (`Runtime::regenerate`).
#[derive(Clone, Copy)]
pub(crate) struct Regen(pub(crate) std::any::TypeId);

/// The state of a `Runtime::regenerate` run on this thread: the payloads
/// made for the reads its node's unbounded run took a payload from, each
/// raised by the first read of its node. Nodes are keyed as a `ReadKey`'s
/// `reading`.
pub(crate) struct Regenerating {
    rt: Option<RuntimeId>,
    made: Vec<(u64, Box<dyn Any + Send>)>,
}

impl Regenerating {
    pub(crate) const fn new() -> Self {
        Self {
            rt: None,
            made: Vec::new(),
        }
    }
}

/// Opens a regenerating run of runtime `rt` whose reads take from `made`.
pub(crate) fn regen_open(rt: RuntimeId, made: Vec<(u64, Box<dyn Any + Send>)>) {
    REGEN.with(|r| *r.borrow_mut() = Regenerating { rt: Some(rt), made });
}

/// Closes the run [`regen_open`] opened: the payloads its reads left.
pub(crate) fn regen_close() -> Vec<(u64, Box<dyn Any + Send>)> {
    REGEN.with(|r| std::mem::replace(&mut *r.borrow_mut(), Regenerating::new()).made)
}

/// The payload made for the open run's read of node `key` of runtime
/// `rt`, if the unbounded run's read took one and no read since raised it.
pub(crate) fn regen_take(rt: RuntimeId, key: u64) -> Option<Box<dyn Any + Send>> {
    REGEN.with(|r| {
        let mut r = r.borrow_mut();
        if r.rt != Some(rt) {
            return None;
        }
        let i = r.made.iter().position(|m| m.0 == key)?;
        Some(r.made.swap_remove(i).1)
    })
}

/// A run's identity across its frame's retries: its node as `(slot,
/// generation)`, and its path and seq, as a `ReadKey` has them.
pub(crate) type RunKey = (u32, u32, u64, u32);

/// What a read took from a compute or a ticket (`note_took`): a copy, or
/// for a payload no copy can stand for, nothing, as a regeneration runs
/// the read node again.
pub(crate) type Took = (u64, Option<Box<dyn Any + Send>>);

/// What each run's reads took (see [`note_took`]), by the read node.
type Takes = HashMap<
    (RuntimeId, RunKey),
    HashMap<u64, Option<Box<dyn Any + Send>>, FxBuildHasher>,
    FxBuildHasher,
>;

fn with_takes<R>(f: impl FnOnce(&mut Takes) -> R) -> R {
    value_code::set_bits(value_code::TOOK_BIT, true);
    TOOK.with(|t| f(&mut t.borrow_mut()))
}

/// The seq and path of the run of node `reading` that ended just above
/// depth `depth`, if it started one.
pub(crate) fn ended_run(rt: RuntimeId, depth: u32, reading: u64) -> Option<(u32, u64)> {
    RUNS.with(|r| {
        (r.borrow().live.get(depth as usize).copied().flatten())
            .filter(|x| {
                x.rt == rt && u64::from(x.frame.0) | (u64::from(x.frame.1) << 32) == reading
            })
            .map(|x| (x.seq, x.path))
    })
}

/// Notes that run `reader`'s read of node `reading` took `payload` from a
/// compute or a ticket, until the run lands (`took_by_run`) or the engine
/// call ends.
pub(crate) fn note_took(rt: RuntimeId, reader: RunKey, reading: u64, payload: &(dyn Any + Send)) {
    let copy = copy_payload(payload).filter(|_| !payload.is::<Regen>());
    with_takes(|t| {
        t.entry((rt, reader)).or_default().insert(reading, copy);
    });
}

/// Whether this thread noted a take in this engine call (`clear` resets
/// it).
#[inline(always)]
pub(crate) fn takes_held() -> bool {
    value_code::thread_word() & value_code::TOOK_BIT != 0
}

/// Whether this thread has noted reads of runtime `rt` no landing took.
pub(crate) fn takes_noted(rt: RuntimeId) -> bool {
    takes_held() && TOOK.with(|t| t.borrow().keys().any(|k| k.0 == rt))
}

/// What run `run`'s reads took, as it lands.
pub(crate) fn took_by_run(rt: RuntimeId, run: RunKey) -> Vec<Took> {
    TOOK.with(|t| t.borrow_mut().remove(&(rt, run)))
        .map_or_else(Vec::new, |took| took.into_iter().collect())
}

/// Runtime `rt`'s nodes claimed by a regenerating run on some thread
/// (`Runtime::rerun`), which another thread's rerun of the same node waits
/// out instead of taking the claim for a compute.
static REGEN_CLAIMS: std::sync::Mutex<Vec<(RuntimeId, u32)>> = std::sync::Mutex::new(Vec::new());

/// Notes or drops `slot` of runtime `rt` as claimed by a regenerating run.
pub(crate) fn regen_claim(rt: RuntimeId, slot: u32, held: bool) {
    let mut c = REGEN_CLAIMS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if held {
        c.push((rt, slot));
    } else if let Some(i) = c.iter().position(|&x| x == (rt, slot)) {
        c.swap_remove(i);
    }
}

/// How many regenerating runs claim `slot` of runtime `rt`.
pub(crate) fn regen_claims(rt: RuntimeId, slot: u32) -> usize {
    REGEN_CLAIMS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|&&x| x == (rt, slot))
        .count()
}

/// Runtime `rt`'s nodes a regenerating run took from Failed to Computing
/// and has not yet put back. Marked and cleared under the runtime's
/// tickets lock with the node's state, so a ticket's holder reads the two
/// as one.
static REGEN_RUNNING: std::sync::Mutex<Vec<(RuntimeId, u32)>> = std::sync::Mutex::new(Vec::new());

/// Marks or clears `slot` of runtime `rt` as rerunning (`REGEN_RUNNING`);
/// clearing an unmarked node does nothing.
pub(crate) fn regen_running(rt: RuntimeId, slot: u32, on: bool) {
    let mut r = REGEN_RUNNING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if on {
        r.push((rt, slot));
    } else if let Some(i) = r.iter().position(|&x| x == (rt, slot)) {
        r.swap_remove(i);
    }
}

/// Whether a regenerating run holds `slot` of runtime `rt` out of Failed.
pub(crate) fn is_regen_running(rt: RuntimeId, slot: u32) -> bool {
    REGEN_RUNNING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(&(rt, slot))
}

/// Whether [`copy_payload`] can copy `payload`.
pub(crate) fn copyable(payload: &(dyn Any + Send)) -> bool {
    payload.is::<crate::cycle::CycleUnwind>()
        || payload.is::<String>()
        || payload.is::<&'static str>()
        || payload.is::<Regen>()
}

/// A copy of a panic payload of a type the engine knows: a cycle unwind's
/// or a `panic!` message. A type added here is added to [`copyable`] too.
pub(crate) fn copy_payload(payload: &(dyn Any + Send)) -> Option<Box<dyn Any + Send>> {
    if let Some(unwind) = payload.downcast_ref::<crate::cycle::CycleUnwind>() {
        return Some(Box::new(crate::cycle::CycleUnwind(unwind.0.clone())));
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return Some(Box::new(message.clone()));
    }
    if let Some(regen) = payload.downcast_ref::<Regen>() {
        return Some(Box::new(*regen));
    }
    payload
        .downcast_ref::<&'static str>()
        .map(|message| Box::new(*message) as Box<dyn Any + Send>)
}

/// Which unwind a panic payload is, for this thread's walk-containment
/// notes and a delivery's [`Delivery::resumed`]: the address of its box,
/// which stays put while the box unwinds. A closure can catch a box and
/// raise a fresh one at the freed address with no engine call between, so
/// the key also holds the type and the text of a `&str` or `String`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PayloadKey {
    /// Every zero-sized payload's box has its type's dangling address, so
    /// only the type tells two of them apart, the engine's own unwinds
    /// included.
    address: usize,
    ty: TypeId,
    /// A hash of the payload's text, 0 for a payload of another type.
    text: u64,
}

pub(crate) fn payload_key(payload: &(dyn Any + Send)) -> PayloadKey {
    let text = |s: &str| BuildHasherDefault::<DefaultHasher>::default().hash_one(s);
    PayloadKey {
        address: std::ptr::from_ref(payload).cast::<()>() as usize,
        ty: Any::type_id(payload),
        text: match payload.downcast_ref::<&'static str>() {
            Some(s) => text(s),
            None => payload.downcast_ref::<String>().map_or(0, |s| text(s)),
        },
    }
}

/// Opens a context, returning its id.
pub(crate) fn open(rt: RuntimeId, boundary: u32, target: Entry, frame_keys: FrameKeys) -> u64 {
    let id = next_id();
    with(|c| {
        let mut x = Context::new(id, rt, boundary, target, frame_keys);
        // The slots above the boundary are now the new context's to keep,
        // as it covers the frames below them.
        if let Some(at) = covering(c, rt, boundary) {
            let i = (boundary - c[at].boundary) as usize;
            x.runs = c[at].runs.split_off(i);
        }
        c.push(x);
    });
    if value_code::thread_word() & value_code::KEPT_BIT != 0 {
        mark_kept();
    }
    id
}

/// The key parts of `rt`'s frames, if `rt` has a context open.
pub(crate) fn frame_keys(rt: RuntimeId) -> Option<FrameKeys> {
    with(|c| c.iter().find(|x| x.rt == rt).map(|x| x.frame_keys))
}

/// Takes the outcome held in `rt`'s contexts for the read `key`, marking
/// every delivery for it taken: a landed value cloned by `clone`, or the
/// unwind (see [`Context::take_unwind`]). Walk directions are not
/// outcomes.
pub(crate) fn take_pending(
    rt: RuntimeId,
    key: &ReadKey,
    clone: impl Fn(&(dyn Any + Send)) -> Box<dyn Any + Send>,
) -> Option<Delivered> {
    with(|c| {
        c.iter_mut().filter(|x| x.rt == rt).find_map(|x| {
            let at = x
                .pending
                .find_keyed(key, |d| !d.taken && !matches!(d.what, Delivered::Walk(..)))?;
            // The read has its outcome, which ended the walk it was
            // blocked in, so a later read under the same key in this
            // run must not resume that walk or recompute its node.
            x.pending.take_keyed(key);
            x.directs.take_keyed(key);
            Some(match &x.pending.get(at).what {
                Delivered::Landed(value, poison) => {
                    Delivered::Landed(Unbuilt::new(clone(value.value())), *poison)
                }
                Delivered::Walk(..) => {
                    unreachable!("incr-core: a walk's direction is not a value")
                }
                Delivered::Unwind { .. } => x.take_unwind(at),
            })
        })
    })
}

/// The taken delivery of a payload no copy can stand for, for read `key`
/// of `rt`'s contexts, was regenerated as the payload `raised`: it counts
/// as the source of the landings the delivery carries, as a copy raised
/// by a take does (`Delivery::resumed`).
pub(crate) fn regenerated(rt: RuntimeId, key: &ReadKey, raised: PayloadKey) {
    with(|c| {
        for x in c.iter_mut().filter(|x| x.rt == rt) {
            for i in x.pending.keyed(key).to_vec() {
                let d = x.pending.get(i);
                if d.taken
                    && matches!(&d.what, Delivered::Unwind { payload, .. } if payload.is::<Regen>())
                {
                    x.pending.resume(i, raised);
                }
            }
        }
    });
}

/// As [`regenerated`], for delivery `i` of context `id`.
pub(crate) fn regenerated_at(id: u64, i: usize, raised: PayloadKey) {
    with(|c| {
        if let Some(x) = c.iter_mut().find(|x| x.id == id) {
            x.pending.resume(i, raised);
        }
    });
}

/// Removes context `id`, if it is still open, returning the deliveries
/// it still holds.
#[must_use]
pub(crate) fn close(id: u64) -> Vec<Delivery> {
    let removed = with(|c| {
        let x = c.remove(c.iter().position(|x| x.id == id)?);
        end_sessions(c, [x.rt]);
        Some(x)
    });
    removed.map(|x| x.pending.into_vec()).unwrap_or_default()
}

/// Removes every context of `rt` whose boundary is at or above `cidx`:
/// a caught unwind has left the reads that opened them.
pub(crate) fn drop_from(rt: RuntimeId, cidx: u32) {
    remove_where(|x| x.rt == rt && x.boundary >= cidx);
}

/// Removes every context, of any runtime, whose boundary is at or above
/// `cidx`: the entry that began there has exited.
pub(crate) fn drop_all_from(cidx: u32) {
    remove_where(|x| x.boundary >= cidx);
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
        end_sessions(c, out.iter().map(|x| x.rt));
        out
    });
    drop(removed);
}

/// Removes every context on the thread: an outermost engine entry
/// starts with none.
#[inline(always)]
pub(crate) fn reset() {
    if value_code::thread_word() & (value_code::REPLAY_BIT | value_code::TOOK_BIT) != 0 {
        clear();
    }
}

#[cold]
#[inline(never)]
fn clear() {
    SUSPENDED.with(|s| s.borrow_mut().clear());
    RUNS.with(|r| *r.borrow_mut() = Runs::default());
    TOOK.with(|t| t.borrow_mut().clear());
    value_code::set_bits(value_code::TOOK_BIT, false);
    set_carry(None);
    let removed = with(std::mem::take);
    drop(removed);
    crate::reserve::end_contexts();
}

/// The innermost context of `rt` whose boundary is at or below `cidx`,
/// as an index into the stack.
pub(crate) fn covering(c: &[Context], rt: RuntimeId, cidx: u32) -> Option<usize> {
    c.iter().rposition(|x| x.rt == rt && x.boundary <= cidx)
}

/// The slot (see [`Context::runs`]) of the runs at position `cidx` along
/// `path`.
fn run_slot(c: &[Context], rt: RuntimeId, cidx: u32, path: u64) -> Option<&RunSlot> {
    let x = &c[covering(c, rt, cidx.checked_sub(1)?)?];
    x.runs.get((cidx - x.boundary - 1) as usize, path)
}

/// As [`run_slot`], made if missing; none when no context covers the frame
/// below, as then no record of a run at `cidx` can outlive its context.
fn run_slot_mut(c: &mut [Context], rt: RuntimeId, cidx: u32, path: u64) -> Option<&mut RunSlot> {
    let x = &mut c[covering(c, rt, cidx.checked_sub(1)?)?];
    Some(x.runs.entry((cidx - x.boundary - 1) as usize, path))
}

/// A run that started while a context of `rt` was open.
#[derive(Clone, Copy)]
struct Run {
    rt: RuntimeId,
    ctl: usize,
    frame: Frame,
    seq: u32,
    path: u64,
    /// When it started, among the thread's runs.
    order: u64,
}

/// The thread's runs (see [`start_run`]), until the last context of their
/// runtime goes.
#[derive(Default)]
struct Runs {
    /// The latest run started at each depth on the thread's stack, from
    /// depth 1; the entries above the stack's top are runs that ended.
    live: Vec<Option<Run>>,
    /// The latest of the runs a later run at their depth replaced while
    /// claims were held, by frame: a member lands at its episode's end,
    /// after the members beside it ran at its depth. Emptied at the next
    /// run's start with no claim held, when every ended run has landed.
    ended: HashMap<(RuntimeId, Frame), Run, FxBuildHasher>,
    next: u64,
}

impl Runs {
    fn forget(&mut self, rt: RuntimeId) {
        for x in &mut self.live {
            if x.is_some_and(|x| x.rt == rt) {
                *x = None;
            }
        }
        while self.live.last().is_some_and(Option::is_none) {
            self.live.pop();
        }
        self.ended.retain(|&(r, _), _| r != rt);
    }
}

/// A run of `frame` along `path` starts, with control block `ctl` at
/// `depth` on the thread's stack: it is given the seq (see
/// [`Context::runs`]) its frame has there now, for its whole run. Only a
/// run under a context of `rt` has one, as a read is replayed only then.
pub(crate) fn start_run(
    rt: RuntimeId,
    ctl: *const crate::cycle::ComputeCtl,
    depth: u32,
    frame: Frame,
    path: u64,
) {
    let seq = with(|c| {
        innermost(c, rt)?;
        Some(run_slot(c, rt, frame.2, path).map_or(0, |s| s.count(frame.0, frame.1)))
    });
    let Some(seq) = seq else {
        return;
    };
    let held = crate::cycle_state::held_any();
    RUNS.with(|r| {
        let r = &mut *r.borrow_mut();
        let at = depth.saturating_sub(1) as usize;
        if held {
            for x in r.live.drain(at.min(r.live.len())..).flatten() {
                let key = (x.rt, x.frame);
                if r.ended.get(&key).is_none_or(|e| e.order < x.order) {
                    r.ended.insert(key, x);
                }
            }
        } else {
            r.live.truncate(at);
            r.ended.clear();
        }
        r.live.resize(at, None);
        r.next += 1;
        r.live.push(Some(Run {
            rt,
            ctl: ctl as usize,
            frame,
            seq,
            path,
            order: r.next,
        }));
    });
}

/// The thread's runs, borrowed to find the seqs of many frames at once.
pub(crate) struct Seqs<'a>(&'a Runs);

impl Seqs<'_> {
    /// The seq of the run of `rt`'s frame with control block `ctl` at
    /// `depth` on the thread's stack, or 0 for one that started with no
    /// context of `rt` open.
    pub(crate) fn of(
        &self,
        rt: RuntimeId,
        ctl: *const crate::cycle::ComputeCtl,
        depth: u32,
    ) -> u32 {
        let Some(at) = depth.checked_sub(1) else {
            return 0;
        };
        (self.0.live.get(at as usize).copied().flatten())
            .filter(|x| x.rt == rt && x.ctl == ctl as usize)
            .map_or(0, |x| x.seq)
    }
}

pub(crate) fn with_seqs<R>(f: impl FnOnce(Seqs<'_>) -> R) -> R {
    RUNS.with(|r| f(Seqs(&r.borrow())))
}

/// As [`Seqs::of`], for one frame.
pub(crate) fn seq_of(rt: RuntimeId, ctl: *const crate::cycle::ComputeCtl, depth: u32) -> u32 {
    with_seqs(|s| s.of(rt, ctl, depth))
}

/// The seq and path of the run of the frame at `depth` whose control block
/// is `ctl`, if it started one.
pub(crate) fn run_at(
    rt: RuntimeId,
    ctl: *const crate::cycle::ComputeCtl,
    depth: u32,
) -> Option<(u32, u64)> {
    RUNS.with(|r| {
        let at = depth.checked_sub(1)? as usize;
        (r.borrow().live.get(at).copied().flatten())
            .filter(|x| x.rt == rt && x.ctl == ctl as usize)
            .map(|x| (x.seq, x.path))
    })
}

/// The seq of the latest run of `rt`'s `frame`, live or ended, and the
/// path it started along, which no frame or virtual frame below it changes
/// while it runs or before its episode's end lands it: for a run that may
/// have left the stack, as a member's that lands at its episode's end. A
/// frame on the stack has them from [`run_at`]. No run gives seq 0 and no
/// path.
pub(crate) fn run_seq(rt: RuntimeId, frame: Frame) -> (u32, Option<u64>) {
    RUNS.with(|r| {
        let r = r.borrow();
        let live = (r.live.iter().rev().flatten()).find(|x| x.rt == rt && x.frame == frame);
        let ended = r.ended.get(&(rt, frame));
        match (live, ended) {
            (Some(l), Some(e)) => Some(if l.order > e.order { l } else { e }),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
        .map_or((0, None), |x| (x.seq, Some(x.path)))
    })
}

/// Drops the runs of each runtime in `gone` that has no context left in
/// `c`: a frame that starts before its next context opens has no run, and
/// must not find one of an earlier frame at its stack address.
fn end_sessions(c: &[Context], gone: impl IntoIterator<Item = RuntimeId>) {
    RUNS.with(|r| {
        let r = &mut *r.borrow_mut();
        if c.is_empty() {
            *r = Runs::default();
            return;
        }
        for rt in gone {
            if innermost(c, rt).is_none() {
                r.forget(rt);
            }
        }
    });
}

/// The runs ended so far above virtual frame `below` in its blocked read,
/// which a `Walk` or `Direct` for that read keeps for the retry.
pub(crate) fn run_snapshot(c: &[Context], rt: RuntimeId, below: &VFrame) -> RunSlot {
    let path = extend_path(below.path, Step::virtual_frame(below));
    run_slot(c, rt, below.cidx + 1, path)
        .cloned()
        .unwrap_or(RunSlot {
            path,
            owner: 0,
            counts: Few::None,
        })
}

/// A read by the frame at `cidx`, at `depth` on the thread's stack,
/// starts, with its frames above along `path`: they count their runs from
/// `from`, what the delivery the read resumes kept, or from none. The
/// reads at `cidx` of frames that were at or above `depth` are over, and
/// of virtual frames no longer there.
pub(crate) fn start_read(rt: RuntimeId, cidx: u32, depth: u32, path: u64, from: Option<RunSlot>) {
    with(|c| {
        let Some(at) = covering(c, rt, cidx) else {
            return;
        };
        let i = (cidx - c[at].boundary) as usize;
        let over = |s: &RunSlot| s.path != path && (s.owner == 0 || s.owner >= depth);
        if c[at].runs.at(i).iter().any(over) {
            let kept: Vec<u64> = (c.iter().filter(|x| x.rt == rt))
                .filter_map(|x| x.vframe_at(cidx))
                .map(|v| extend_path(v.path, Step::virtual_frame(v)))
                .collect();
            c[at]
                .runs
                .at_mut(i)
                .retain(|s| !over(s) || kept.contains(&s.path));
        }
        let slot = c[at].runs.entry(i, path);
        slot.owner = depth;
        match from.filter(|f| f.path == path) {
            Some(f) => slot.counts = f.counts,
            None => slot.counts = Few::None,
        }
    });
}

/// The run of `frame` along `path` ended, not doomed, so a later run there
/// gets the next seq. It is counted in its read's slot, made if missing,
/// and in each untaken `Walk` or `Direct` kept for a virtual frame's read
/// at the position below, whose retry resumes after it. A speculative run
/// is no run of the unbounded run's and counts nowhere.
pub(crate) fn end_run(rt: RuntimeId, frame: Frame, path: u64) {
    if speculative() {
        return;
    }
    let (node, generation, cidx) = frame;
    with(|c| {
        if let Some(below) = cidx.checked_sub(1) {
            for x in c.iter_mut().filter(|x| x.rt == rt) {
                let Some(key) = x.vframe_at(below).map(ReadKey::blocked) else {
                    continue;
                };
                for j in 0..x.pending.keyed(&key).len() {
                    let d = x.pending.get_mut(x.pending.keyed(&key)[j]);
                    if !d.taken && d.runs.path == path && matches!(d.what, Delivered::Walk(..)) {
                        d.runs.bump(node, generation);
                    }
                }
                x.directs.bump(&key, path, node, generation);
            }
        }
        if let Some(slot) = run_slot_mut(c, rt, cidx, path) {
            slot.bump(node, generation);
        }
    });
}

/// A run at position `cidx` that is not a driver entry's, a member run,
/// ended. The deliveries for the frames its run made are read again by its
/// retry if a doom discarded it, and returned to be dropped otherwise, as
/// the driver does for an entry's run.
#[must_use]
pub(crate) fn run_ended(rt: RuntimeId, cidx: u32, doomed: bool) -> Vec<Delivery> {
    with(|c| {
        let Some(at) = covering(c, rt, cidx) else {
            return Vec::new();
        };
        let x = &mut c[at];
        let level = cidx - x.boundary;
        if doomed {
            for i in x.pending.of_maker(level) {
                x.pending.set_taken(i, false);
            }
            x.directs.rearm_maker(level);
            return Vec::new();
        }
        x.end_maker(level)
    })
}

/// The earliest value noted for each doomed frame, as in
/// [`Context::claim_revs`] and [`Context::run_failures`], indexed by frame,
/// as `(node, generation, cidx)`, and by maker: a long chain dooms and
/// retries every frame.
#[derive(Default)]
pub(crate) struct EarliestByFrame {
    /// Each frame's records, as `(path, maker, value)`.
    by_frame: Index<Frame, (u64, u32, u64)>,
    by_maker: Index<u32, Frame>,
    /// The positions of each node's frames, by node and generation.
    by_node: Index<(u32, u32), u32>,
    /// The nodes whose runs a release ended, until a doomed run of the node
    /// notes a record again: a record taken for one meanwhile would be one
    /// of the ended runs'.
    #[cfg(debug_assertions)]
    ended: HashSet<(u32, u32), FxBuildHasher>,
}

/// A frame's node, generation and position.
pub(crate) type Frame = (u32, u32, u32);

impl EarliestByFrame {
    fn note(&mut self, frame: Frame, path: u64, maker: u32, value: u64) {
        #[cfg(debug_assertions)]
        self.ended.remove(&(frame.0, frame.1));
        match self.by_frame.entry(frame) {
            MapEntry::Vacant(e) => {
                e.insert(Few::One((path, maker, value)));
                list(&mut self.by_node, (frame.0, frame.1), frame.2);
            }
            MapEntry::Occupied(mut e) => {
                let records = e.get_mut();
                match records.as_mut_slice().iter_mut().find(|r| r.0 == path) {
                    Some(r) => {
                        r.2 = r.2.min(value);
                        return;
                    }
                    None => records.push((path, maker, value)),
                }
            }
        }
        list(&mut self.by_maker, maker, frame);
    }

    pub(crate) fn has(&self, frame: Frame) -> bool {
        self.by_frame.contains_key(&frame)
    }

    fn has_node(&self, node: u32, generation: u32) -> bool {
        self.by_node.contains_key(&(node, generation))
    }

    pub(crate) fn take(&mut self, frame: Frame, path: u64) -> Option<u64> {
        let value = self.remove(frame, |r| r.0 == path);
        #[cfg(debug_assertions)]
        assert!(
            value.is_none() || !self.ended.contains(&(frame.0, frame.1)),
            "incr-core: a run took a record of a run a release ended"
        );
        value
    }

    /// Removes the frame's records that `f` accepts, returning the value of
    /// the last.
    fn remove(&mut self, frame: Frame, f: impl Fn(&(u64, u32, u64)) -> bool) -> Option<u64> {
        let records = self.by_frame.get_mut(&frame)?;
        let mut value = None;
        let empty = records.retain(|r| {
            let gone = f(r);
            if gone {
                value = Some(r.2);
            }
            !gone
        });
        if empty {
            self.by_frame.remove(&frame);
            unlist(&mut self.by_node, &(frame.0, frame.1), |&c| c == frame.2);
        }
        value
    }

    /// The run of the frame at `frame` along `path` ended.
    fn end_frame(&mut self, frame: Frame, path: u64) {
        self.remove(frame, |r| r.0 == path);
    }

    /// A release ended the runs of every frame of `node` at `generation`.
    fn end_node(&mut self, node: u32, generation: u32) {
        #[cfg(debug_assertions)]
        self.ended.insert((node, generation));
        for cidx in listed(&self.by_node, &(node, generation)) {
            self.remove((node, generation, cidx), |_| true);
        }
    }

    fn end_maker(&mut self, level: u32) {
        let frames = self.by_maker.remove(&level);
        for &frame in frames.as_ref().map_or(&[][..], Few::as_slice) {
            self.remove(frame, |r| r.1 == level);
        }
    }
}

/// The doomed run of `node` at `cidx`, whose virtual frame is on a
/// context's stack, claimed at walked watermark `rev`.
pub(crate) fn note_claim_rev(rt: RuntimeId, node: u32, generation: u32, cidx: u32, rev: u64) {
    note_earliest(rt, node, generation, cidx, rev, |x| &mut x.claim_revs);
}

/// The doomed member run of `node` at `cidx`, whose virtual frame is on a
/// context's stack, began at thread failure count `failures`.
pub(crate) fn note_run_failures(
    rt: RuntimeId,
    node: u32,
    generation: u32,
    cidx: u32,
    failures: u64,
) {
    note_earliest(rt, node, generation, cidx, failures, |x| {
        &mut x.run_failures
    });
}

fn note_earliest(
    rt: RuntimeId,
    node: u32,
    generation: u32,
    cidx: u32,
    value: u64,
    records: fn(&mut Context) -> &mut EarliestByFrame,
) {
    with(|c| {
        for x in c.iter_mut().filter(|x| x.rt == rt) {
            let Some(v) = cidx
                .checked_sub(x.boundary)
                .and_then(|pos| x.vstack.get(pos as usize))
                .filter(|v| v.node == node && v.cidx == cidx)
            else {
                continue;
            };
            let (path, maker) = (v.path, v.maker);
            records(x).note((node, generation, cidx), path, maker, value);
            return;
        }
    });
}

/// The innermost context of `rt`, as an index into the stack.
pub(crate) fn innermost(c: &[Context], rt: RuntimeId) -> Option<usize> {
    c.iter().rposition(|x| x.rt == rt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notes::Notes;

    fn keyed_run_slot(key: &ReadKey) -> RunSlot {
        let step = Step {
            node: key.node,
            episode: key.episode,
            reading: key.reading,
            ordinal: key.ordinal,
            seq: key.seq,
        };
        RunSlot {
            path: extend_path(key.path, step),
            ..RunSlot::default()
        }
    }

    fn direct_key(node: u32, reading: u64) -> ReadKey {
        ReadKey {
            node,
            generation: 0,
            cidx: 5,
            reading,
            ordinal: 1,
            path: 3,
            seq: 0,
            episode: Some(9),
            foreign: None,
        }
    }

    /// A released node's directs are gone at once though they leave
    /// their positions only with their maker, and a direct delivered
    /// again after the release counts afresh.
    #[test]
    fn a_released_nodes_directs_are_gone_and_come_back_fresh() {
        let (x, y) = (direct_key(1, 7), direct_key(2, 7 | 1 << 32));
        let mut directs = Directs::new(0);
        directs.deliver(x, 0, keyed_run_slot(&x));
        directs.deliver(y, 0, keyed_run_slot(&y));
        let (px, py) = (keyed_run_slot(&x).path, keyed_run_slot(&y).path);
        directs.bump(&x, px, 7, 0);
        directs.bump(&x, px, 8, 0);
        directs.bump(&y, py, 7, 1);
        directs.remove_node(1, 0);
        assert!(!directs.frame_awaits((1, 0, 5)));
        assert!(directs.frame_awaits((2, 0, 5)));
        assert!(directs.take(&x).is_none());
        directs.deliver(x, 0, keyed_run_slot(&x));
        directs.bump(&x, px, 8, 0);
        let fresh = directs.take(&x).expect("delivered again");
        assert_eq!(
            (fresh.path, fresh.count(7, 0), fresh.count(8, 0)),
            (px, 0, 1)
        );
        let y_runs = directs.take(&y).expect("not released");
        assert_eq!((y_runs.count(7, 1), y_runs.count(7, 0)), (1, 0));
        directs.remove_maker(0);
        assert!(directs.released.is_empty() && directs.more_runs.is_empty());
    }

    /// A shallow context's directs make only the positions it uses.
    #[test]
    #[cfg(not(loom))]
    fn a_shallow_direct_makes_few_positions() {
        let key = direct_key(1, 7);
        let mut directs = Directs::new(0);
        take_note_work();
        directs.deliver(key, 0, keyed_run_slot(&key));
        let units = take_note_work();
        assert!(
            units <= 8,
            "{units} units of note work for a direct at position 5"
        );
    }

    /// Running out of stamps drops the released directs and keeps the
    /// others' counts.
    #[test]
    fn running_out_of_stamps_keeps_the_live_directs_runs() {
        let (x, y, z) = (direct_key(1, 7), direct_key(2, 7), direct_key(3, 7));
        let mut directs = Directs::new(0);
        directs.deliver(x, 0, keyed_run_slot(&x));
        directs.deliver(y, 1, keyed_run_slot(&y));
        directs.bump(&y, keyed_run_slot(&y).path, 8, 0);
        directs.remove_node(1, 0);
        directs.next_stamp = u32::MAX;
        directs.deliver(z, 1, keyed_run_slot(&z));
        assert_eq!(directs.live, 2);
        assert!(directs.released.is_empty());
        assert_eq!(directs.take(&y).expect("kept").count(8, 0), 1);
        assert!(directs.take(&z).is_some() && directs.take(&x).is_none());
        directs.remove_maker(0);
        directs.remove_maker(1);
        assert_eq!(directs.live, 0);
    }

    /// A deep direct that goes with its maker's run and comes back with the
    /// next doom is held at its position again without making the
    /// positions below it again.
    #[test]
    #[cfg(not(loom))]
    fn a_deep_direct_held_again_makes_no_positions_again() {
        let depth = 100_000;
        let key = ReadKey {
            node: 1,
            generation: 0,
            cidx: depth,
            reading: 2,
            ordinal: 1,
            path: 3,
            seq: 0,
            episode: None,
            foreign: None,
        };
        let mut directs = Directs::new(0);
        take_note_work();
        for _ in 0..10 {
            directs.deliver(key, 0, keyed_run_slot(&key));
            directs.remove_maker(0);
        }
        let units = take_note_work();
        assert!(
            units < 2 * u64::from(depth),
            "{units} units of note work for 10 holds of a direct at position {depth}"
        );
    }

    /// A cut that takes a node's newer, deeper note and keeps its older one
    /// leaves the older stamp: the frame's containment names only the
    /// landing its kept notes name.
    #[test]
    fn a_cut_newer_note_leaves_the_older_stamp() {
        let rt = RuntimeId::for_test(7);
        let payload: Box<dyn Any + Send> = Box::new("p");
        let key = payload_key(&*payload);
        let mut notes = Notes::new();
        notes.push(rt, key, 1, ((5, 0), 10), false);
        notes.push(rt, key, 3, ((5, 0), 20), false);
        let whole = notes.copy(rt, key, 0).into_nodes();
        assert_eq!(whole, vec![((5, 0), 1, 10), ((5, 0), 3, 20)]);
        let mut cut = notes.copy(rt, key, 0);
        let Nodes::Shared { cut: c, .. } = &mut cut.nodes else {
            panic!("a copy of a log shares it");
        };
        *c = Some(Cut {
            depth: 2,
            seen: 0,
            landed: Rc::new(CutLandings::default()),
        });
        assert_eq!(cut.into_nodes(), vec![((5, 0), 1, 10)]);
    }

    /// Copy `a` of a log is taken before a removal and copy `b` after it,
    /// and `b` is marked first: `a` still holds the removed note, so its
    /// mark must not stop at the watermark `b` left.
    #[test]
    fn an_earlier_copy_marks_the_notes_a_later_copy_left_out() {
        for marks in [Marks::Kept, Marks::Copied] {
            let rt = RuntimeId::for_test(7);
            let payload: Box<dyn Any + Send> = Box::new("p");
            let key = payload_key(&*payload);
            let mut notes = Notes::new();
            notes.push(rt, key, 1, ((1, 0), 0), false);
            notes.push(rt, key, 2, ((2, 0), 0), false);
            let a = notes.copy(rt, key, 0);
            notes.forget_node(rt, (2, 0));
            notes.push(rt, key, 3, ((3, 0), 0), false);
            let b = notes.copy(rt, key, 0);
            mark_in(marks, rt, &b);
            mark_in(marks, rt, &a);
            let set = match marks {
                Marks::Kept => &KEPT_HELD,
                Marks::Copied => &COPIED,
            };
            assert!(set.with(|s| s.borrow().contains(&(rt, (2, 0)))));
        }
    }

    /// A pending list holding one passed delivery of the frame at 3 whose
    /// payload is keyed `source`, with landings `nodes`.
    fn passed_at_3(source: PayloadKey, nodes: Vec<((u32, u32), u32)>) -> Pending {
        let read = ReadKey {
            node: 1,
            generation: 0,
            cidx: 3,
            reading: 2,
            ordinal: 1,
            path: 0,
            seq: 0,
            episode: None,
            foreign: None,
        };
        let what = Delivered::Unwind {
            cycle: false,
            poison: crate::cycle::POISONED,
            attributed: false,
            payload: Box::new("p"),
            nodes: Landings::owned(
                Some(source),
                nodes.into_iter().map(|(n, d)| (n, d, 0)).collect(),
            ),
        };
        let mut pending = Pending::default();
        pending.push(Delivery::new(read, 0, what));
        pending.pass(0);
        pending
    }

    fn holds(pending: &mut Pending, node: (u32, u32)) -> bool {
        let Delivered::Unwind { nodes, .. } = &mut pending.get_mut(0).what else {
            unreachable!()
        };
        nodes.holds(node)
    }

    static P: &str = "p";
    static Q: &str = "q";

    fn key(payload: &'static &'static str) -> PayloadKey {
        payload_key(payload)
    }

    /// A passed delivery's landing just above its frame is a note at the
    /// position of the frame's next read, which a walk's entry there
    /// containing the same payload retires.
    #[test]
    fn a_containment_just_above_a_frame_retires_its_passed_landing() {
        let mut pending = passed_at_3(key(&P), vec![((2, 0), 0)]);
        let contained = |payload| Retire::Contained { cidx: 4, payload };
        pending.retire(&contained(key(&Q)), false);
        assert!(holds(&mut pending, (2, 0)));
        pending.retire(&contained(key(&P)), false);
        assert!(!holds(&mut pending, (2, 0)));
    }

    /// The copy a take raises stands for the delivery's payload as its
    /// source does, so a containment of the copy retires the landing too.
    #[test]
    fn a_containment_of_a_resumed_copy_retires_its_passed_landing() {
        let mut pending = passed_at_3(key(&P), vec![((2, 0), 0)]);
        pending.resume(0, key(&Q));
        pending.retire(
            &Retire::Contained {
                cidx: 4,
                payload: key(&Q),
            },
            false,
        );
        assert!(!holds(&mut pending, (2, 0)));
    }

    /// A copy renoted in two parts, the second reusing the first, notes
    /// what the whole copy renoted at once does: once the deeper frame's
    /// notes go, both keep the shallower note's own stamp, as the frames
    /// that made the notes would.
    #[cfg(not(loom))]
    #[test]
    fn a_renote_reused_across_a_stamp_change_notes_what_a_whole_one_does() {
        let rt = RuntimeId::for_test(7);
        let source: Box<dyn Any + Send> = Box::new("s");
        let s = payload_key(&*source);
        let mut from = Notes::new();
        from.push(rt, s, 1, ((5, 0), 10), false);
        let early = from.copy(rt, s, 0);
        from.push(rt, s, 3, ((5, 0), 20), false);
        let late = from.copy(rt, s, 0);
        let mut reused = Notes::new();
        reused.note_copy(rt, key(&P), 0, early.as_shared().unwrap());
        take_note_work();
        reused.note_copy(rt, key(&P), 0, late.as_shared().unwrap());
        let reusing = take_note_work();
        let mut whole = Notes::new();
        whole.note_copy(rt, key(&Q), 0, late.as_shared().unwrap());
        assert!(
            reusing < take_note_work(),
            "the second renote reused the first"
        );
        reused.forget_from(2, None);
        whole.forget_from(2, None);
        let reused = reused.copy(rt, key(&P), 0).into_nodes();
        let whole = whole.copy(rt, key(&Q), 0).into_nodes();
        assert_eq!(reused, whole);
        assert_eq!(whole, vec![((5, 0), 1, 10)]);
    }

    /// A passed delivery's payload regenerated as `Q` counts as its source,
    /// so a containment of `Q` just above its frame reaches its landings.
    #[test]
    fn a_containment_of_a_regenerated_payload_retires_its_passed_landing() {
        unsafe fn parts(_: *const (), _: u32, _: u32, _: bool) -> KeyParts {
            (0, 0, None, 0)
        }
        let target = Entry {
            node: 0,
            generation: 0,
            suspect: false,
            spec: false,
            retry: Retry::Walk,
            walked: false,
            r5: false,
        };
        let keys = FrameKeys {
            owner: std::ptr::null(),
            parts,
        };
        let id = open(RuntimeId::for_test(7), 0, target, keys);
        let context = |c: &mut Vec<Context>| c.iter_mut().position(|x| x.id == id).unwrap();
        with(|c| {
            let at = context(c);
            c[at].pending = passed_at_3(key(&P), vec![((2, 0), 0)]);
        });
        regenerated_at(id, 0, key(&Q));
        let held = with(|c| {
            let at = context(c);
            let pending = &mut c[at].pending;
            pending.retire(
                &Retire::Contained {
                    cidx: 4,
                    payload: key(&Q),
                },
                false,
            );
            holds(pending, (2, 0))
        });
        drop(close(id));
        assert!(!held);
    }

    /// A delivery that loses its deeper landings is filed again as one
    /// whose landings sit just above its frame, which a containment there
    /// of its payload still reaches.
    #[test]
    fn a_delivery_left_just_above_its_frame_is_reached_there() {
        let mut pending = passed_at_3(key(&P), vec![((2, 0), 0), ((5, 0), 1)]);
        pending.retire(&Retire::Above(4), false);
        assert!(!holds(&mut pending, (5, 0)));
        assert!(pending.by_top.is_empty());
        assert!(matches!(pending.filed.get(&0), Some(Filed::Shallow(4, _))));
        pending.retire(
            &Retire::Contained {
                cidx: 4,
                payload: key(&P),
            },
            false,
        );
        assert!(!holds(&mut pending, (2, 0)));
    }

    /// A later take's copy replaces the earlier one's as the payload a
    /// containment must name.
    #[test]
    fn a_later_take_replaces_the_copy_a_containment_retires() {
        static R: &str = "r";
        let mut pending = passed_at_3(key(&P), vec![((2, 0), 0)]);
        pending.resume(0, key(&Q));
        pending.resume(0, key(&R));
        assert!(!pending.by_payload.contains(&(4, key(&Q), 0)));
        let contained = |payload| Retire::Contained { cidx: 4, payload };
        pending.retire(&contained(key(&Q)), false);
        assert!(holds(&mut pending, (2, 0)));
        pending.retire(&contained(key(&R)), false);
        assert!(!holds(&mut pending, (2, 0)));
    }

    /// A copy keyed as the source shares its entry, which a later take's
    /// copy leaves to the source.
    #[test]
    fn a_take_after_a_copy_keyed_as_the_source_keeps_the_source_listed() {
        let mut pending = passed_at_3(key(&P), vec![((2, 0), 0)]);
        pending.resume(0, key(&P));
        pending.resume(0, key(&Q));
        assert!(pending.by_payload.contains(&(4, key(&P), 0)));
        pending.retire(
            &Retire::Contained {
                cidx: 4,
                payload: key(&P),
            },
            false,
        );
        assert!(!holds(&mut pending, (2, 0)));
    }

    /// Opens a context holding `pending`, runs `f` on it and closes it.
    fn in_context(pending: Pending, f: impl FnOnce(u64)) {
        unsafe fn no_parts(_: *const (), _: u32, _: u32, _: bool) -> KeyParts {
            unreachable!()
        }
        let entry = Entry {
            node: 1,
            generation: 0,
            suspect: false,
            spec: false,
            retry: Retry::Compute,
            walked: false,
            r5: false,
        };
        let keys = FrameKeys {
            owner: std::ptr::null(),
            parts: no_parts,
        };
        let id = open(RuntimeId::SENTINEL, 1, entry, keys);
        with(|c| c.iter_mut().find(|x| x.id == id).unwrap().pending = pending);
        f(id);
        drop(close(id));
    }

    fn retires_contained_at_4(id: u64, payload: PayloadKey) -> bool {
        with(|c| {
            let x = c.iter_mut().find(|x| x.id == id).unwrap();
            x.pending
                .retire(&Retire::Contained { cidx: 4, payload }, false);
            !holds(&mut x.pending, (2, 0))
        })
    }

    /// A regenerated payload stands for a delivery's source as a take's
    /// copy does, so a containment of it retires a landing just above the
    /// frame.
    #[test]
    fn a_containment_of_a_payload_regenerated_at_its_delivery_retires_its_landing() {
        in_context(passed_at_3(key(&P), vec![((2, 0), 0)]), |id| {
            regenerated_at(id, 0, key(&Q));
            assert!(retires_contained_at_4(id, key(&Q)));
        });
    }

    #[test]
    fn a_containment_of_a_payload_regenerated_for_its_read_retires_its_landing() {
        let mut pending = passed_at_3(key(&P), vec![((2, 0), 0)]);
        let d = pending.get_mut(0);
        d.taken = true;
        let Delivered::Unwind { payload, .. } = &mut d.what else {
            unreachable!()
        };
        *payload = Box::new(Regen(std::any::TypeId::of::<&str>()));
        let read = d.key;
        in_context(pending, |id| {
            regenerated(RuntimeId::SENTINEL, &read, key(&Q));
            assert!(retires_contained_at_4(id, key(&Q)));
        });
    }

    /// Short of a node a copy holds, a landing of another payload retires
    /// only the landings above its position.
    #[test]
    fn a_landing_retires_passed_landings_only_above_it() {
        let mut pending = passed_at_3(key(&P), vec![((2, 0), 0), ((5, 0), 1)]);
        pending.retire(
            &Retire::Landing {
                cidx: 4,
                payload: Some(key(&Q)),
                node: (9, 0),
            },
            false,
        );
        assert!(holds(&mut pending, (2, 0)));
        assert!(!holds(&mut pending, (5, 0)));
    }
}
