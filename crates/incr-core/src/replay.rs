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
use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
use std::rc::Rc;

use crate::cycle_state::{held_episode, Episode, UnwindKind};
use crate::deferred::Unbuilt;
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
    /// For a member run, the episode its member is in: a discovery run of
    /// the node at the position, and the runs of another episode, are other
    /// runs.
    pub(crate) episode: Option<u64>,
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
            episode: v.member_of,
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
}

/// A context's deliveries. Each keeps its index until it is removed, and
/// is found by its read, the level whose run made it, its frame, or the
/// provenance record it carries, without a pass over the others: a deep
/// chain's context holds a delivery per level, and its reads look them up
/// per level.
#[derive(Default)]
pub(crate) struct Pending {
    slots: Vec<Option<Delivery>>,
    live: usize,
    by_key: HashMap<ReadKey, Vec<usize>>,
    by_maker: HashMap<u32, Vec<usize>>,
    by_frame: HashMap<(u32, u32, u32), Vec<usize>>,
    traced: Vec<usize>,
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
        if d.provenance.is_some() {
            self.traced.push(i);
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
        if d.provenance.is_some() {
            self.traced.retain(|&j| j != i);
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
        nodes: Vec<(u32, u32)>,
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
    /// the unbounded run's walk goes on after that compute.
    Walk(Box<crate::runtime::Walk>),
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
            pending: Pending::default(),
            spec_base: target.spec,
            claims: Vec::new(),
            episodes: Vec::new(),
            memo: Vec::new(),
            resume: None,
            direct: None,
            lows: Vec::new(),
            drop_at: None,
            drop_point: None,
            provenance: None,
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
    /// text is another panic. The path tells this frame's delivery from
    /// another run's at the position, whose taken copy's freed box can hold
    /// a fresh copy of this one.
    pub(crate) fn retaken_landing(
        &mut self,
        (node, generation, cidx, path): (u32, u32, u32, u64),
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
    /// where a doom stopped it, for the frame's retry.
    pub(crate) fn deliver_walk(&mut self, below: &VFrame, walk: crate::runtime::Walk) {
        let key = ReadKey::blocked(below);
        self.pending
            .remove_keyed(&key, |d| matches!(d.what, Delivered::Walk(_)));
        self.pending.push(Delivery {
            key,
            maker: below.maker,
            taken: false,
            blocked: false,
            what: Delivered::Walk(Box::new(walk)),
            provenance: None,
            below: Vec::new(),
            resumed: None,
            landed: None,
        });
    }

    /// Sends the read virtual frame `below` was blocked in to its compute
    /// in the retry of the entry that made the frame. A pass at a higher
    /// level can pop the frame above it, and that pass's end must not drop
    /// the delivery before the frame's retry reads it.
    pub(crate) fn deliver_direct(&mut self, below: &VFrame) {
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

    /// Truncates `vstack` to `len` with no Tarjan pops, returning the
    /// episodes suspended at the frames removed: they are abandoned, and
    /// their members are released with the claims held there.
    pub(crate) fn truncate_vstack(&mut self, len: usize) -> Vec<Episode> {
        let mut abandoned = Vec::new();
        while self.vstack.len() > len {
            if let Some(ep) = self.pop_vframe().and_then(|v| v.ep) {
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
    pub(crate) what: Delivered,
}

#[cfg(not(loom))]
thread_local! {
    static KEPT_READS: RefCell<Vec<KeptRead>> = const { RefCell::new(Vec::new()) };
    static CONTEXTS: RefCell<Vec<Context>> = const { RefCell::new(Vec::new()) };
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
    static SPECULATIVE: Cell<u32> = const { Cell::new(0) };
    static CARRY: Cell<Option<Carry>> = const { Cell::new(None) };
    static MEMBER_RUNS: RefCell<Vec<(u32, (u64, u32))>> = const { RefCell::new(Vec::new()) };
    static SUSPENDED: RefCell<HashMap<u64, (u64, u32)>> = RefCell::new(HashMap::new());
}

#[cfg(loom)]
loom::thread_local! {
    static KEPT_READS: RefCell<Vec<KeptRead>> = RefCell::new(Vec::new());
    static CONTEXTS: RefCell<Vec<Context>> = RefCell::new(Vec::new());
    static NEXT_ID: Cell<u64> = Cell::new(1);
    static SPECULATIVE: Cell<u32> = Cell::new(0);
    static CARRY: Cell<Option<Carry>> = Cell::new(None);
    static MEMBER_RUNS: RefCell<Vec<(u32, (u64, u32))>> = RefCell::new(Vec::new());
    static SUSPENDED: RefCell<HashMap<u64, (u64, u32)>> = RefCell::new(HashMap::new());
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
        out
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
        }
    }

    /// The step of the frame `v` stands for.
    pub(crate) fn virtual_frame(v: &VFrame) -> Self {
        Self {
            node: v.node,
            episode: v.member_of,
            reading: v.reading,
            ordinal: v.ordinal,
        }
    }
}

/// The path of a frame above one at `path` that is in `step`.
pub(crate) fn extend_path(path: u64, step: Step) -> u64 {
    [
        u64::from(step.node),
        step.episode.map_or(0, |e| e + 1),
        step.reading,
        u64::from(step.ordinal) + 1,
    ]
    .into_iter()
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
pub(crate) fn open(rt: RuntimeId, boundary: u32, target: Entry) -> u64 {
    let id = next_id();
    with(|c| c.push(Context::new(id, rt, boundary, target)));
    id
}

/// Removes context `id`, if it is still open, returning the deliveries
/// it still holds.
#[must_use]
pub(crate) fn close(id: u64) -> Vec<Delivery> {
    let removed = with(|c| c.iter().position(|x| x.id == id).map(|at| c.remove(at)));
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
    set_carry(None);
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
        x.pending.extract_maker(level)
    })
}

/// The innermost context of `rt`, as an index into the stack.
pub(crate) fn innermost(c: &[Context], rt: RuntimeId) -> Option<usize> {
    c.iter().rposition(|x| x.rt == rt)
}
