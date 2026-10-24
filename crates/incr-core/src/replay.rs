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
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
use std::rc::Rc;

use crate::cycle_state::{held_episode, Episode, UnwindKind};
use crate::deferred::Unbuilt;
use crate::fx::FxBuildHasher;
use crate::handle::RuntimeId;
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
    /// the same retry is an ordinary one, as in the unbounded run.
    pub(crate) taken: bool,
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
/// are found by position and path alone (see [`Context::runs`]). `owner` is the reading
/// frame's depth on the thread's stack, which only decides how long
/// [`start_read`] keeps the slot; it is 0 for a slot no read set, which a
/// run's end makes under a virtual frame's blocked read or the opener's.
#[derive(Clone, Default)]
pub(crate) struct RunSlot {
    pub(crate) path: u64,
    pub(crate) owner: u32,
    pub(crate) counts: Vec<(u32, u32, u32)>,
}

impl RunSlot {
    fn count(&self, node: u32, generation: u32) -> u32 {
        self.counts
            .iter()
            .find(|&&(n, g, _)| (n, g) == (node, generation))
            .map_or(0, |&(_, _, k)| k)
    }

    fn bump(&mut self, node: u32, generation: u32) {
        match self
            .counts
            .iter_mut()
            .find(|(n, g, _)| (*n, *g) == (node, generation))
        {
            Some((_, _, k)) => *k += 1,
            None => self.counts.push((node, generation, 1)),
        }
    }
}

/// The Failed landings noted for a payload that a copy of it carries, each
/// with its depth above the reading frame, and the payload they were noted
/// for.
#[derive(Clone, Default)]
pub(crate) struct Landings {
    pub(crate) source: Option<PayloadKey>,
    pub(crate) nodes: Vec<((u32, u32), u32)>,
}

impl Landings {
    fn is_deep(&self) -> bool {
        self.nodes.iter().any(|&(_, depth)| depth > 0)
    }

    fn max_depth(&self) -> u32 {
        self.nodes
            .iter()
            .map(|&(_, depth)| depth)
            .max()
            .unwrap_or(0)
    }

    /// Applies `ev` to the landings of a read by the frame at `cidx`, as it
    /// applies to the notes they stand for; `resumed` is the copy the
    /// frame's latest take raised, which counts as the source.
    fn retire(&mut self, cidx: u32, ev: &Retire, resumed: Option<PayloadKey>) {
        let ours = |p: PayloadKey| self.source == Some(p) || resumed == Some(p);
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
        self.nodes.retain(|&(n, depth)| !gone(n, cidx + 1 + depth));
    }
}

/// An event that retires notes (`runtime_contain`), applied to the copies
/// of them that kept reads and passed deliveries hold.
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
    by_key: HashMap<ReadKey, Vec<usize>>,
    by_maker: HashMap<u32, Vec<usize>>,
    by_frame: HashMap<(u32, u32, u32), Vec<usize>>,
    /// By the frame's node and generation, at any position.
    by_node: HashMap<(u32, u32), Vec<usize>>,
    traced: Vec<usize>,
    /// The passed deliveries that carry a landing deeper than just above
    /// their frame, which a pop can retire.
    deep: Vec<usize>,
}

fn frame_of(key: &ReadKey) -> (u32, u32, u32) {
    (key.node, key.generation, key.cidx)
}

/// Drops index `i` from the list `map` holds under `at`.
fn unlist<K: std::hash::Hash + Eq>(map: &mut HashMap<K, Vec<usize>>, at: &K, i: usize) {
    if let Some(list) = map.get_mut(at) {
        list.retain(|&j| j != i);
        if list.is_empty() {
            map.remove(at);
        }
    }
}

impl Pending {
    pub(crate) fn push(&mut self, d: Delivery) {
        let i = self.slots.len();
        self.by_key.entry(d.key).or_default().push(i);
        self.by_maker.entry(d.maker).or_default().push(i);
        self.by_frame.entry(frame_of(&d.key)).or_default().push(i);
        self.by_node
            .entry((d.key.node, d.key.generation))
            .or_default()
            .push(i);
        if d.provenance.is_some() {
            self.traced.push(i);
        }
        if d.passed && d.is_deep() {
            self.deep.push(i);
        }
        self.slots.push(Some(d));
        self.live += 1;
    }

    pub(crate) fn remove(&mut self, i: usize) -> Delivery {
        let d = self.slots[i]
            .take()
            .expect("incr-core: a delivery is removed once");
        unlist(&mut self.by_key, &d.key, i);
        unlist(&mut self.by_maker, &d.maker, i);
        unlist(&mut self.by_frame, &frame_of(&d.key), i);
        unlist(&mut self.by_node, &(d.key.node, d.key.generation), i);
        if d.provenance.is_some() {
            self.traced.retain(|&j| j != i);
        }
        if d.passed {
            self.deep.retain(|&j| j != i);
        }
        self.live -= 1;
        // No index is held across a point where none is live.
        if self.live == 0 {
            self.slots.clear();
        }
        d
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
            if d.is_deep() {
                self.deep.push(i);
            }
        }
    }

    /// Applies `ev` to the landings the passed deliveries carry: those of
    /// every passed delivery for a node's landing that a copy may hold,
    /// else only the deeper ones, since an event at or below a frame
    /// ends that frame's run.
    fn retire(&mut self, ev: &Retire, all: bool) {
        let slots = &mut self.slots;
        let apply = |d: &mut Delivery| {
            if d.key.cidx < ev.cidx() || all {
                let resumed = d.resumed;
                if let Delivered::Unwind { nodes, .. } = &mut d.what {
                    nodes.retire(d.key.cidx, ev, resumed);
                }
            }
        };
        if all {
            for d in slots.iter_mut().flatten().filter(|d| d.passed) {
                apply(d);
            }
        }
        self.deep.retain(|&i| {
            let d = slots[i]
                .as_mut()
                .expect("incr-core: a delivery's index is live");
            if !all {
                apply(d);
            }
            d.is_deep()
        });
    }

    /// The deliveries for read `key`, oldest first.
    pub(crate) fn keyed(&self, key: &ReadKey) -> &[usize] {
        self.by_key.get(key).map_or(&[], Vec::as_slice)
    }

    /// The oldest delivery for read `key` that `f` accepts.
    pub(crate) fn find_keyed(&self, key: &ReadKey, f: impl Fn(&Delivery) -> bool) -> Option<usize> {
        self.keyed(key).iter().copied().find(|&i| f(self.get(i)))
    }

    /// Marks every delivery for read `key` taken.
    pub(crate) fn take_keyed(&mut self, key: &ReadKey) {
        for i in self.keyed(key).to_vec() {
            self.get_mut(i).taken = true;
        }
    }

    /// Removes the deliveries for read `key` that `f` accepts.
    pub(crate) fn remove_keyed(&mut self, key: &ReadKey, f: impl Fn(&Delivery) -> bool) {
        for i in self.keyed(key).to_vec() {
            if f(self.get(i)) {
                self.remove(i);
            }
        }
    }

    /// The deliveries the run at `level` made, oldest first.
    pub(crate) fn of_maker(&self, level: u32) -> Vec<usize> {
        self.by_maker.get(&level).cloned().unwrap_or_default()
    }

    /// Removes the deliveries the run at `level` made, oldest first.
    pub(crate) fn extract_maker(&mut self, level: u32) -> Vec<Delivery> {
        self.of_maker(level)
            .into_iter()
            .map(|i| self.remove(i))
            .collect()
    }

    /// The deliveries for reads by the frame of `node` at `generation` and
    /// `cidx`, oldest first.
    pub(crate) fn of_frame(&self, node: u32, generation: u32, cidx: u32) -> Vec<usize> {
        self.by_frame
            .get(&(node, generation, cidx))
            .cloned()
            .unwrap_or_default()
    }

    /// Removes the deliveries for reads by that frame, oldest first.
    pub(crate) fn extract_frame(&mut self, node: u32, generation: u32, cidx: u32) -> Vec<Delivery> {
        self.of_frame(node, generation, cidx)
            .into_iter()
            .map(|i| self.remove(i))
            .collect()
    }

    /// Removes, oldest first, the deliveries for reads by the frames of
    /// `node` at `generation`, at any position.
    pub(crate) fn extract_node(&mut self, node: u32, generation: u32) -> Vec<Delivery> {
        self.by_node
            .get(&(node, generation))
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|i| self.remove(i))
            .collect()
    }

    /// The deliveries that carry a provenance record, oldest first.
    pub(crate) fn traced(&self) -> &[usize] {
        &self.traced
    }

    pub(crate) fn into_vec(self) -> Vec<Delivery> {
        self.slots.into_iter().flatten().collect()
    }

    #[cfg(all(test, not(loom)))]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Delivery> {
        self.slots.iter().flatten()
    }
}

impl Delivery {
    fn is_deep(&self) -> bool {
        matches!(&self.what, Delivered::Unwind { nodes, .. } if nodes.is_deep())
    }

    fn is_untaken_panic(&self) -> bool {
        !self.taken && matches!(self.what, Delivered::Unwind { cycle: false, .. })
    }
}

/// A delivery's hold on its runtime's provenance record, which stays
/// live while some delivery holds it.
pub(crate) trait Provenance {
    fn record(&self) -> u64;
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
    /// payload is the `CycleUnwind`, or a user panic. It poisons the
    /// reader unless it is the cycle of a node that landed `Cyclic`, which
    /// a read of the node raises with the node landed.
    Unwind {
        cycle: bool,
        poison: bool,
        /// A member was marked failed in the run the panic came out of,
        /// so a member run that takes it again is not.
        attributed: bool,
        payload: Box<dyn Any + Send>,
        /// The landings the panic made that a walk's entry may contain
        /// with it (`runtime_provenance`), which follow each copy taken.
        nodes: Landings,
    },
    /// The entry landed without a Clean value. The unbounded run's read
    /// takes the slot as the compute left it, poisoned, where a later read
    /// would walk and run the node again. It is that value, cloned when
    /// the delivery is made, because a later run in the context may
    /// compute the node again before the retry reads it. Until a read
    /// takes it, it is a clone the unbounded run never builds.
    Landed(Unbuilt<dyn Any + Send>),
    /// The rest of the walk of the frame's blocked read, from the compute
    /// a doom stopped it at: the retry's walk of that read resumes it, as
    /// the unbounded run's walk goes on after that compute. With the node
    /// of the entry the stop pushed back, which the context's run of that
    /// compute pops: a doomed entry is not pushed back, so a pending entry
    /// of its node on top is a repeat the walk still owes.
    Walk(Box<crate::runtime::Walk>, Option<u32>),
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
    pub(crate) pending: Pending,
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
    /// position reads none of the first's deliveries. A read sets its slot as it starts, empty
    /// or to what the delivery it resumes kept, so a retry counts as the
    /// run it redoes did. Keyed by `(cidx, path)`.
    pub(crate) runs: BTreeMap<(u32, u64), RunSlot>,
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
            vstack: Vec::new(),
            vindex: HashMap::new(),
            running: None,
            running_entry: None,
            pending: Pending::default(),
            spec_base: target.spec,
            claims: Vec::new(),
            episodes: Vec::new(),
            memo: Vec::new(),
            resume: None,
            direct: None,
            lows: Vec::new(),
            claim_revs: EarliestByFrame::default(),
            run_failures: EarliestByFrame::default(),
            runs: BTreeMap::new(),
            drop_at: None,
            drop_point: None,
            provenance: None,
            frame_keys,
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
        self.levels[level as usize].pop_front();
        self.trim_levels();
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
    }

    /// The highest virtual frame below position `cidx`.
    pub(crate) fn vframe_below(&self, cidx: u32) -> Option<&VFrame> {
        let above = cidx.saturating_sub(self.boundary) as usize;
        self.vstack[..above.min(self.vstack.len())].last()
    }

    /// Delivers `what` to the read virtual frame `below` was blocked in.
    pub(crate) fn deliver_blocked(&mut self, below: &VFrame, what: Delivered) {
        self.push_blocked(below, what, None);
    }

    /// Delivers the unwind out of the running frame at `cidx` to the
    /// read of [`Self::unwind_target`], with the provenance record
    /// published for that read.
    pub(crate) fn deliver_unwind(&mut self, cidx: u32, what: Delivered) {
        let below = self
            .unwind_target(cidx)
            .expect("incr-core: an unwind is delivered to a virtual frame")
            .clone();
        let key = ReadKey::blocked(&below);
        let provenance = self
            .provenance
            .take()
            .and_then(|(k, p)| (k == key).then_some(p));
        self.push_blocked(&below, what, provenance);
    }

    fn push_blocked(
        &mut self,
        below: &VFrame,
        what: Delivered,
        provenance: Option<Rc<dyn Provenance>>,
    ) {
        let key = ReadKey::blocked(below);
        self.pending.remove_keyed(&key, |d| d.blocked);
        self.pending.push(Delivery {
            key,
            maker: below.maker,
            taken: false,
            blocked: true,
            what,
            provenance,
            below: Vec::new(),
            resumed: None,
            landed: None,
            passed: false,
            runs: RunSlot::default(),
        });
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
        self.pending.traced().iter().filter_map(move |&i| {
            let d = self.pending.get(i);
            let k = d.key;
            let p = d.provenance.as_ref()?;
            (d.is_untaken_panic()
                && ((k.cidx == at && k.node == node && k.generation == generation)
                    || d.below.contains(&(at, node, generation))))
            .then(|| (i, p.record()))
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
    /// for a retry doomed again; a payload that can't be copied is
    /// delivered once.
    pub(crate) fn take_unwind(&mut self, i: usize) -> Delivered {
        let key = self.pending.get(i).key;
        self.pending.take_keyed(&key);
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
            return self.pending.remove(i).what;
        };
        let what = Delivered::Unwind {
            cycle: *cycle,
            poison: *poison,
            attributed: *attributed,
            payload: copy,
            nodes: nodes.clone(),
        };
        if let Delivered::Unwind { payload, .. } = &what {
            self.pending.get_mut(i).resumed = Some(payload_key(&**payload));
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
        self.pending
            .remove_keyed(&key, |d| matches!(d.what, Delivered::Walk(..)));
        self.pending.push(Delivery {
            key,
            maker: below.maker,
            taken: false,
            blocked: false,
            what: Delivered::Walk(Box::new(walk), pushed),
            provenance: None,
            below: Vec::new(),
            resumed: None,
            landed: None,
            passed: false,
            runs,
        });
    }

    /// Sends the read virtual frame `below` was blocked in to its compute
    /// in the retry of the entry that made the frame. A pass at a higher
    /// level can pop the frame above it, and that pass's end must not drop
    /// the delivery before the frame's retry reads it. A kept one keeps its
    /// `runs`: the retry that took it ran none of the read's walk.
    pub(crate) fn deliver_direct(&mut self, below: &VFrame, runs: RunSlot) {
        let key = ReadKey::blocked(below);
        if self
            .pending
            .find_keyed(&key, |d| matches!(d.what, Delivered::Direct))
            .is_none()
        {
            self.pending.push(Delivery {
                key,
                maker: below.maker,
                taken: false,
                blocked: false,
                what: Delivered::Direct,
                provenance: None,
                below: Vec::new(),
                resumed: None,
                landed: None,
                passed: false,
                runs,
            });
        }
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

#[cfg(not(loom))]
thread_local! {
    static KEPT_READS: RefCell<Vec<KeptRead>> = const { RefCell::new(Vec::new()) };
    static KEPT_DEPTH: Cell<u32> = const { Cell::new(0) };
    static COPIED: RefCell<[u64; 64]> = const { RefCell::new([0; 64]) };
    static CONTEXTS: RefCell<Vec<Context>> = const { RefCell::new(Vec::new()) };
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
    static SPECULATIVE: Cell<u32> = const { Cell::new(0) };
    static CARRY: Cell<Option<Carry>> = const { Cell::new(None) };
    static MEMBER_RUNS: RefCell<Vec<(u32, (u64, u32))>> = const { RefCell::new(Vec::new()) };
    static SUSPENDED: RefCell<HashMap<u64, (u64, u32)>> = RefCell::new(HashMap::new());
    static RUNS: RefCell<Runs> = RefCell::new(Runs::default());
}

#[cfg(loom)]
loom::thread_local! {
    static KEPT_READS: RefCell<Vec<KeptRead>> = RefCell::new(Vec::new());
    static KEPT_DEPTH: Cell<u32> = Cell::new(0);
    static COPIED: RefCell<[u64; 64]> = RefCell::new([0; 64]);
    static CONTEXTS: RefCell<Vec<Context>> = RefCell::new(Vec::new());
    static NEXT_ID: Cell<u64> = Cell::new(1);
    static SPECULATIVE: Cell<u32> = Cell::new(0);
    static CARRY: Cell<Option<Carry>> = Cell::new(None);
    static MEMBER_RUNS: RefCell<Vec<(u32, (u64, u32))>> = RefCell::new(Vec::new());
    static SUSPENDED: RefCell<HashMap<u64, (u64, u32)>> = RefCell::new(HashMap::new());
    static RUNS: RefCell<Runs> = RefCell::new(Runs::default());
}

#[cfg(all(test, not(loom)))]
thread_local! {
    static VINDEX_REPUSHES: Cell<u64> = const { Cell::new(0) };
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

/// Keeps `read` for its frame's retry. Reads of frames that have popped go:
/// every frame at or above the reading one's position, but the reading
/// one, which is a run of its own if this is its first such read.
pub(crate) fn keep_read(read: KeptRead, fresh: bool) {
    if let Delivered::Unwind { nodes, .. } = &read.what {
        KEPT_DEPTH.with(|k| k.set(k.get().max(nodes.max_depth())));
    }
    let gone: Vec<KeptRead> = KEPT_READS.with(|d| {
        let mut d = d.borrow_mut();
        let gone = d
            .extract_if(.., |e| {
                e.rt == read.rt
                    && e.cidx >= read.cidx
                    && !(e.cidx == read.cidx && e.ctl == read.ctl && !fresh)
            })
            .collect();
        d.push(read);
        gone
    });
    value_code::set_kept_bit(true);
    drop(gone);
}

fn copied_bit(rt: RuntimeId, node: (u32, u32)) -> (usize, u64) {
    let h = BuildHasherDefault::<DefaultHasher>::default().hash_one((rt, node)) as usize;
    ((h >> 6) & 63, 1 << (h & 63))
}

/// Marks `nodes` as landings a copy may hold, so that a later landing of
/// one of them finds the copies to drop it from.
pub(crate) fn mark_copied(rt: RuntimeId, nodes: &Landings) {
    COPIED.with(|c| {
        let mut c = c.borrow_mut();
        for &(n, _) in &nodes.nodes {
            let (w, b) = copied_bit(rt, n);
            c[w] |= b;
        }
    });
}

/// Whether a kept read or a context's delivery may hold copied landings.
pub(crate) fn copies_held() -> bool {
    value_code::thread_word() & (value_code::KEPT_BIT | value_code::REPLAY_BIT) != 0
}

/// Applies `ev`, an event that retires notes of runtime `rt`, to the
/// copies of them kept reads and the contexts' passed deliveries hold,
/// outside a speculative run.
pub(crate) fn retire_copies(rt: RuntimeId, ev: &Retire) {
    let all = ev.node().is_some_and(|n| {
        let (w, b) = copied_bit(rt, n);
        COPIED.with(|c| c.borrow()[w] & b != 0)
    });
    let at = ev.cidx();
    KEPT_READS.with(|d| {
        let mut d = d.borrow_mut();
        let deepest = KEPT_DEPTH.with(Cell::get);
        // Kept reads are pushed by the stack's top frame after those at or
        // above it go, and go when their frame pops, so they run in frame
        // order.
        for e in d.iter_mut().rev() {
            if !all && e.cidx + 1 + deepest <= at {
                break;
            }
            if e.rt == rt && (e.cidx < at || all) {
                if let Delivered::Unwind { nodes, .. } = &mut e.what {
                    nodes.retire(e.cidx, ev, None);
                }
            }
        }
        if d.is_empty() {
            KEPT_DEPTH.with(|k| k.set(0));
        }
    });
    with(|c| {
        for x in c.iter_mut().filter(|x| x.rt == rt) {
            x.pending.retire(ev, all);
        }
    });
}

/// The reads kept for the run of control block `ctl` at `cidx`.
pub(crate) fn take_kept_reads(
    rt: RuntimeId,
    cidx: u32,
    ctl: *const crate::cycle::ComputeCtl,
) -> Vec<KeptRead> {
    extract_kept(|e| e.rt == rt && e.cidx == cidx && e.ctl == ctl)
}

/// Drops the reads kept for frames of `rt` at or above `cidx`, which a
/// frame pop at `cidx` ends: only a doom takes them, and it took them as
/// it marked the frames.
#[cold]
#[inline(never)]
pub(crate) fn drop_kept_from(rt: RuntimeId, cidx: u32) {
    drop(extract_kept(|e| e.rt == rt && e.cidx >= cidx));
}

/// Taken out of the list before they drop, since a kept value's `Drop`
/// runs user code.
fn extract_kept(mut which: impl FnMut(&KeptRead) -> bool) -> Vec<KeptRead> {
    KEPT_READS.with(|d| {
        let mut d = d.borrow_mut();
        let out = d.extract_if(.., |e| which(e)).collect();
        value_code::set_kept_bit(!d.is_empty());
        if d.is_empty() {
            KEPT_DEPTH.with(|k| k.set(0));
            if value_code::thread_word() & value_code::REPLAY_BIT == 0 {
                COPIED.with(|c| *c.borrow_mut() = [0; 64]);
            }
        }
        out
    })
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
    KEPT_READS.with(|d| {
        let earlier = d
            .borrow()
            .iter()
            .filter(|e| {
                e.rt == rt
                    && e.cidx == cidx
                    && e.ctl == ctl
                    && e.reading == reading
                    && e.foreign == Some(foreign)
            })
            .count();
        earlier as u32 + 1
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

/// The Failed read a provenance record named, until a later Failed read
/// of the same node on this thread or the end of the outermost entry.
pub(crate) fn carry() -> Option<Carry> {
    CARRY.with(Cell::get)
}

pub(crate) fn set_carry(carry: Option<Carry>) {
    CARRY.with(|c| c.set(carry));
}

/// A copy of a panic payload of a type the engine knows: a cycle unwind's
/// or a `panic!` message.
pub(crate) fn copy_payload(payload: &(dyn Any + Send)) -> Option<Box<dyn Any + Send>> {
    if let Some(unwind) = payload.downcast_ref::<crate::cycle::CycleUnwind>() {
        return Some(Box::new(crate::cycle::CycleUnwind(unwind.0.clone())));
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return Some(Box::new(message.clone()));
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
#[derive(Clone, Copy, PartialEq, Eq)]
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
            x.runs = c[at].runs.split_off(&(boundary + 1, 0));
        }
        c.push(x);
    });
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
            let at = x.pending.find_keyed(key, |d| {
                !d.taken && !matches!(d.what, Delivered::Walk(..) | Delivered::Direct)
            })?;
            // The read has its outcome, which ended the walk it was
            // blocked in, so a later read under the same key in this
            // run must not resume that walk or recompute its node.
            x.pending.take_keyed(key);
            Some(match &x.pending.get(at).what {
                Delivered::Landed(value) => Delivered::Landed(Unbuilt::new(clone(value.value()))),
                Delivered::Walk(..) | Delivered::Direct => {
                    unreachable!("incr-core: a walk's direction is not a value")
                }
                Delivered::Unwind { .. } => x.take_unwind(at),
            })
        })
    })
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
    if value_code::thread_word() & value_code::REPLAY_BIT != 0 {
        clear();
    }
}

#[cold]
#[inline(never)]
fn clear() {
    SUSPENDED.with(|s| s.borrow_mut().clear());
    RUNS.with(|r| *r.borrow_mut() = Runs::default());
    set_carry(None);
    let removed = with(std::mem::take);
    drop(removed);
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
    x.runs.get(&(cidx, path))
}

/// As [`run_slot`], made if missing; none when no context covers the frame
/// below, as then no record of a run at `cidx` can outlive its context.
fn run_slot_mut(c: &mut [Context], rt: RuntimeId, cidx: u32, path: u64) -> Option<&mut RunSlot> {
    let at = covering(c, rt, cidx.checked_sub(1)?)?;
    Some(c[at].runs.entry((cidx, path)).or_insert_with(|| RunSlot {
        path,
        owner: 0,
        counts: Vec::new(),
    }))
}

/// A run that started while a context of `rt` was open.
#[derive(Clone, Copy)]
struct Run {
    rt: RuntimeId,
    ctl: usize,
    frame: Frame,
    seq: u32,
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

/// The seq of the latest run of `rt`'s `frame`, live or ended: for a run
/// that may have left the stack, as a member's that lands at its episode's
/// end. A frame on the stack has its seq from [`seq_of`].
pub(crate) fn run_seq(rt: RuntimeId, frame: Frame) -> u32 {
    RUNS.with(|r| {
        let r = r.borrow();
        let live = (r.live.iter().rev().flatten()).find(|x| x.rt == rt && x.frame == frame);
        let ended = r.ended.get(&(rt, frame));
        match (live, ended) {
            (Some(l), Some(e)) => if l.order > e.order { l } else { e }.seq,
            (Some(x), None) | (None, Some(x)) => x.seq,
            (None, None) => 0,
        }
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
            counts: Vec::new(),
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
        let over: Vec<_> = (c[at].runs.range((cidx + 1, 0)..=(cidx + 1, u64::MAX)))
            .filter(|(_, s)| {
                s.path != path
                    && (s.owner == 0 || s.owner >= depth)
                    && !(c.iter().filter(|x| x.rt == rt))
                        .filter_map(|x| x.vframe_at(cidx))
                        .any(|v| extend_path(v.path, Step::virtual_frame(v)) == s.path)
            })
            .map(|(&k, _)| k)
            .collect();
        for k in over {
            c[at].runs.remove(&k);
        }
        let slot =
            run_slot_mut(c, rt, cidx + 1, path).expect("incr-core: the context covers the read");
        slot.owner = depth;
        match from.filter(|f| f.path == path) {
            Some(f) => slot.counts = f.counts,
            None => slot.counts.clear(),
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
                    if !d.taken
                        && d.runs.path == path
                        && matches!(d.what, Delivered::Walk(..) | Delivered::Direct)
                    {
                        d.runs.bump(node, generation);
                    }
                }
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
                x.pending.get_mut(i).taken = false;
            }
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
    by_frame: HashMap<Frame, Vec<(u64, u32, u64)>>,
    by_maker: HashMap<u32, Vec<Frame>>,
    /// The positions of each node's frames, by node and generation.
    by_node: HashMap<(u32, u32), Vec<u32>>,
    /// The nodes whose runs a release ended, until a doomed run of the node
    /// notes a record again: a record taken for one meanwhile would be one
    /// of the ended runs'.
    #[cfg(debug_assertions)]
    ended: std::collections::HashSet<(u32, u32)>,
}

/// A frame's node, generation and position.
pub(crate) type Frame = (u32, u32, u32);

impl EarliestByFrame {
    fn note(&mut self, frame: Frame, path: u64, maker: u32, value: u64) {
        #[cfg(debug_assertions)]
        self.ended.remove(&(frame.0, frame.1));
        let records = self.by_frame.entry(frame).or_default();
        if records.is_empty() {
            self.by_node
                .entry((frame.0, frame.1))
                .or_default()
                .push(frame.2);
        }
        match records.iter_mut().find(|r| r.0 == path) {
            Some(r) => r.2 = r.2.min(value),
            None => {
                records.push((path, maker, value));
                self.by_maker.entry(maker).or_default().push(frame);
            }
        }
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
        records.retain(|r| {
            let gone = f(r);
            if gone {
                value = Some(r.2);
            }
            !gone
        });
        if records.is_empty() {
            self.by_frame.remove(&frame);
            let key = (frame.0, frame.1);
            if let Some(at) = self.by_node.get_mut(&key) {
                at.retain(|&c| c != frame.2);
                if at.is_empty() {
                    self.by_node.remove(&key);
                }
            }
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
        for cidx in self
            .by_node
            .get(&(node, generation))
            .cloned()
            .unwrap_or_default()
        {
            self.remove((node, generation, cidx), |_| true);
        }
    }

    fn end_maker(&mut self, level: u32) {
        for frame in self.by_maker.remove(&level).unwrap_or_default() {
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
