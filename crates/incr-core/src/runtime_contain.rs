//! A walk's containment of what its cleaning raises. The dirty walk
//! cleans every recorded dep of a node before rerunning it, so it can
//! compute a dep the rerun no longer reads. A non-root walk entry whose
//! compute panics or closes a cycle therefore lands as a changed dep and
//! the walk goes on; a panic's payload stays with the Failed landings it
//! made, as a ticket, for the first read of any of them.
//!
//! The entry's frame decides where it ends in the unwind, before its
//! claims are released: the closure's panic arm, or the abort of the
//! episode the entry roots. A contained unwind lands as any other and
//! skips what an unwind tells the reading frame; where it stops, in the
//! panic arm or at the episode's boundary, its landings become the ticket.
//!
//! The landings an unwind makes above the containing entry are found
//! through notes this thread keeps per landing, keyed by the address of
//! the panic's box. A note lives only while that box unwinds: every point
//! an unwind stops retires the notes above it, and a box the engine moves
//! into a delivery or drops takes its notes with it, so no later box at a
//! reused address meets them. A closure can still catch a box and raise a
//! fresh one at the freed address with no engine call between, so the key
//! also holds the text of a `&str` or `String` payload.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Mutex, PoisonError};

use super::*;
use crate::locks::Lock;
use crate::notes::Notes;
use crate::replay::{payload_key, Landings, Retire};

/// A Failed read's node left Failed before the read could tell which
/// landing it saw.
pub(super) struct Left;

/// The tickets every thread's walks hold.
#[derive(Default)]
pub(crate) struct Tickets {
    next: u64,
    /// By the order they were held in, so that a node's flip to Dirty
    /// removes it only from tickets of landings before the flip.
    held: BTreeMap<u64, Ticket>,
    /// The ticket of each landing a held ticket keeps, which is one at
    /// most: a landing joins a ticket only if it is in none.
    landings: BTreeMap<(u32, u32), u64>,
    /// The tickets of deleted nodes and of nodes a set's walk flipped,
    /// dropped once the delete, sweep or walk that released them holds no
    /// lock and has finished.
    buried: Vec<Ticket>,
}

/// A contained panic, held while any of `nodes`, the `(node, generation)`
/// landings it failed at or above the entry's position, stays in that
/// Failed landing. A node's later Failed landing, its flip to Dirty or its
/// delete removes it.
struct Ticket {
    nodes: Vec<(u32, u32)>,
    /// The mutex only makes the table `Sync`: the payload moves in and out
    /// under the write lock.
    payload: Mutex<Box<dyn Any + Send>>,
}

pub(crate) fn tickets<C: Cells>() -> C::Lock<Tickets> {
    <C::Lock<Tickets> as Lock<Tickets>>::new(Tickets::default())
}

/// An episode's abort that the walk's entry rooting it at `cidx` contains,
/// on its way to the episode's boundary.
pub(super) struct Contained {
    cidx: u32,
    payload: Box<dyn Any + Send>,
}

#[cfg(not(loom))]
thread_local! {
    static NOTES: RefCell<Notes> = const { RefCell::new(Notes::new()) };
    static WALKED_EPISODE: Cell<(u32, u32)> = const { Cell::new((u32::MAX, 0)) };
}

#[cfg(loom)]
loom::thread_local! {
    static NOTES: RefCell<Notes> = RefCell::new(Notes::new());
    static WALKED_EPISODE: Cell<(u32, u32)> = Cell::new((u32::MAX, 0));
}

/// The position and root slot of the innermost running episode whose
/// root is a walk's entry, while it runs.
pub(super) struct WalkedEpisode((u32, u32));

impl WalkedEpisode {
    pub(super) fn enter(cidx: u32, root: NodeId) -> Self {
        Self(WALKED_EPISODE.with(|w| w.replace((cidx, root.0))))
    }
}

impl Drop for WalkedEpisode {
    fn drop(&mut self) {
        WALKED_EPISODE.with(|w| w.set(self.0));
    }
}

/// The root of the episode at `cidx`, if it is a walk's entry.
pub(super) fn walked_episode(cidx: u32) -> Option<NodeId> {
    WALKED_EPISODE.with(|w| {
        let (at, root) = w.get();
        (at == cidx).then_some(NodeId(root))
    })
}

/// Edits this thread's notes and keeps `value_code::NOTES_BIT` in step.
fn with_notes<R>(f: impl FnOnce(&mut Notes) -> R) -> R {
    NOTES.with(|n| {
        let mut n = n.borrow_mut();
        let held = !n.is_empty();
        let r = f(&mut n);
        let holds = !n.is_empty();
        if held != holds {
            value_code::set_notes_bit(holds);
        }
        r
    })
}

fn notes_held() -> bool {
    value_code::thread_word() & value_code::NOTES_BIT != 0
}

/// Removes the landings in `landings` from the tickets before `mark` (all
/// of them for `u64::MAX`), returning the payloads left with no landing,
/// which the caller drops outside the lock: a drop is user code.
fn release(t: &mut Tickets, mark: u64, landings: RangeInclusive<(u32, u32)>) -> Vec<Ticket> {
    let found: Vec<((u32, u32), u64)> = t
        .landings
        .range(landings)
        .filter(|&(_, &seq)| seq < mark)
        .map(|(&landing, &seq)| (landing, seq))
        .collect();
    let mut stale = Vec::new();
    for (landing, seq) in found {
        t.landings.remove(&landing);
        let c = t
            .held
            .get_mut(&seq)
            .expect("incr-core: an indexed landing's ticket is held");
        c.nodes.retain(|&n| n != landing);
        if c.nodes.is_empty() {
            stale.extend(t.held.remove(&seq));
        }
    }
    stale
}

/// Every generation of node `slot`.
fn slot_landings(slot: u32) -> RangeInclusive<(u32, u32)> {
    (slot, 0)..=(slot, u32::MAX)
}

/// Hands `stale` to `bag`, which drops it with no lock held and a panic
/// of its `Drop` handled as any bag item's.
fn defer<C: Cells>(stale: Vec<Ticket>, bag: &mut Deferred<C>) {
    if !stale.is_empty() {
        bag.push_erased(Box::new(stale));
    }
}

impl<C: Cells> Runtime<C> {
    /// Whether any ticket may be held. Set under the lock before a node's
    /// Failed store, so a reader that sees the node Failed sees it.
    fn tickets_held(&self) -> bool {
        self.tickets_held.load(Relaxed)
    }

    /// Before a write guard on the tickets drops.
    fn sync_held(&self, t: &Tickets) {
        self.tickets_held
            .store(!t.held.is_empty() || !t.buried.is_empty(), Relaxed);
    }

    /// The payload a walk contained for `id`'s current Failed landing, taken
    /// by the first read of any node it failed.
    ///
    /// The ticket's nodes are noted against the payload just above the
    /// reading frame: a read from a compute whose panic a walk's entry
    /// below contains again is no reader's observation, so that entry
    /// holds them too. A speculative run's read takes none: a closure in
    /// the run can catch and drop the payload before the run's discard
    /// would hold it again, and the real run's read is owed it.
    pub(super) fn take_contained(&self, id: NodeId) -> Result<Option<Box<dyn Any + Send>>, Left> {
        if !self.tickets_held() || replay::speculative() {
            return Ok(None);
        }
        let node = self.nodes.get(id.0);
        let mut t = self.tickets.write();
        // Every Failed landing removes the node from the tickets before its
        // store, so under the lock a node still Failed at this generation
        // is in the latest landing's ticket, if any.
        let landing = (id.0, node.generation());
        if state::load::<C>(node.state_cell()) != NodeState::Failed {
            return Err(Left);
        }
        let Some(seq) = t.landings.get(&landing).copied() else {
            return Ok(None);
        };
        let Ticket { nodes, payload } = t
            .held
            .remove(&seq)
            .expect("incr-core: an indexed landing's ticket is held");
        for n in &nodes {
            t.landings.remove(n);
        }
        self.sync_held(&t);
        drop(t);
        let payload = payload.into_inner().unwrap_or_else(PoisonError::into_inner);
        self.note_read(nodes, &*payload, true);
        Ok(Some(payload))
    }

    /// Before `id` lands Failed at frame `cidx`, from `land_failed`: drops
    /// `id` from the tickets of earlier landings and notes `payload`, the
    /// panic unwinding, for a walk's entry below to contain. Returns the
    /// mark that orders this landing after every ticket it removed from, or
    /// 0 with none held.
    #[cold]
    #[inline(never)]
    pub(super) fn failed_landing(
        &self,
        id: NodeId,
        generation: u32,
        cidx: u32,
        payload: Option<&(dyn Any + Send)>,
        regenerable: bool,
        bag: &mut Deferred<C>,
    ) -> u64 {
        let rt = self.runtime_id;
        let landing = (id.0, generation);
        self.note_failed_run(id, generation, cidx, regenerable);
        if let Some(payload) = payload {
            // The frames above `cidx` belong to this unwind, so an earlier
            // unwind's notes there are spent. Notes at `cidx` are the reading
            // frame's, whose closure may still hold the payload they follow.
            let at = payload_key(payload);
            with_notes(|n| {
                n.forget_from(cidx + 1, Some(at));
                // This landing replaces the node's earlier Failed landings,
                // so no payload noted for them may hold the node again.
                n.forget_node(rt, landing);
                n.push(rt, at, cidx, landing, false);
            });
        } else if notes_held() {
            with_notes(|n| n.forget_node(rt, landing));
        }
        self.retire_copies(&Retire::Landing {
            cidx,
            payload: payload.map(payload_key),
            node: landing,
        });
        if !self.tickets_held() {
            return 0;
        }
        let (stale, mark) = {
            let mut t = self.tickets.write();
            let stale = release(&mut t, u64::MAX, landing..=landing);
            t.next += 1;
            self.sync_held(&t);
            (stale, t.next)
        };
        defer(stale, bag);
        mark
    }

    /// The episode at `cidx`, rooted at a walk's entry, aborts with
    /// `payload`, which the entry contains: the unwind goes on, past what
    /// it would tell the reading frame, to the episode's boundary. What the
    /// abort released stays in the episode's bag: the containment ends the
    /// unwind, so the bag drains in normal mode and a panicking `Drop` in
    /// it reaches the read.
    #[cold]
    #[inline(never)]
    pub(super) fn raise_contained(&self, cidx: u32, payload: Box<dyn Any + Send>) -> ! {
        std::panic::resume_unwind(Box::new(Contained { cidx, payload }))
    }

    /// An unwind out of the rounds of an episode rooted at a walk's entry:
    /// a [`Contained`] stops here, anything else unwinds on.
    #[cold]
    #[inline(never)]
    pub(super) fn contain_unwind(&self, payload: Box<dyn Any + Send>, bag: &mut Deferred<C>) {
        let c = match payload.downcast::<Contained>() {
            Ok(c) => c,
            Err(payload) => std::panic::resume_unwind(payload),
        };
        self.contain(c.cidx, c.payload, bag);
    }

    /// Stops an unwind a walk's entry at `cidx` contains: the notes above
    /// `cidx` and those of `payload` at it are spent, and the Failed
    /// landings `payload` made there hold it for their first read. Another
    /// payload's notes at `cidx` are the reading frame's, as at a landing.
    pub(super) fn contain(&self, cidx: u32, payload: Box<dyn Any + Send>, bag: &mut Deferred<C>) {
        let (rt, at) = (self.runtime_id, payload_key(&*payload));
        self.retire_copies(&Retire::Contained { cidx, payload: at });
        let nodes = if notes_held() {
            with_notes(|n| n.contain(rt, cidx, at))
        } else {
            Vec::new()
        };
        if let Some(payload) = self.hold(nodes, payload) {
            bag.push_erased(payload);
        }
    }

    /// Holds `payload` for those of `nodes` still in their Failed landing
    /// and in no other ticket, or returns it when none is.
    fn hold(
        &self,
        mut nodes: Vec<(u32, u32)>,
        payload: Box<dyn Any + Send>,
    ) -> Option<Box<dyn Any + Send>> {
        if nodes.is_empty() {
            return Some(payload);
        }
        let mut t = self.tickets.write();
        nodes.retain(|&(slot, generation)| {
            let node = self.nodes.get(slot);
            node.generation() == generation
                && state::load::<C>(node.state_cell()) == NodeState::Failed
                && !t.landings.contains_key(&(slot, generation))
        });
        if nodes.is_empty() {
            return Some(payload);
        }
        let seq = t.next;
        t.next += 1;
        for &n in &nodes {
            t.landings.insert(n, seq);
        }
        t.held.insert(
            seq,
            Ticket {
                nodes,
                payload: Mutex::new(payload),
            },
        );
        self.sync_held(&t);
        None
    }

    /// A set's walk flips `id` from Failed to Dirty, under the tickets'
    /// lock: a `hold` of the landing either precedes the flip, which then
    /// releases its ticket, or sees the node Dirty and holds nothing. The
    /// payloads it alone kept wait with the buried ones until the walk has
    /// flagged every dependent. False if the state moved first.
    #[cold]
    #[inline(never)]
    pub(super) fn flip_failed(&self, id: NodeId) -> bool {
        self.flip_failed_from(id, NodeState::Failed)
    }

    /// [`Self::flip_failed`] for a node in state `from`: a regenerating
    /// run's node is Computing over its Failed landing.
    #[cold]
    #[inline(never)]
    pub(super) fn flip_failed_from(&self, id: NodeId, from: NodeState) -> bool {
        let node = self.nodes.get(id.0);
        // Read while the node is Failed, when no run can land a new record.
        let token = self.failed_run_token(id.0, node.generation());
        let mut t = self.tickets.write();
        if state::try_transition::<C>(node.state_cell(), from, NodeState::Dirty).is_err() {
            return false;
        }
        let stale = release(&mut t, u64::MAX, slot_landings(id.0));
        t.buried.extend(stale);
        self.sync_held(&t);
        drop(t);
        self.park(ParkPoint::FlippedFailed(id));
        self.forget_failed_run_if(id.0, node.generation(), token);
        true
    }

    /// After a set's walk: drops the tickets its flips released.
    #[inline(always)]
    pub(super) fn drop_released_tickets(&self) {
        if self.tickets_held() {
            self.drop_released();
        }
    }

    #[cold]
    #[inline(never)]
    fn drop_released(&self) {
        if let Some(payload) = self.drop_buried() {
            std::panic::resume_unwind(payload);
        }
    }

    /// `id`'s own compute landed it Dirty past the walk that would have
    /// flipped it, leaving the Failed landing that ordered before `mark`:
    /// the payloads join the compute's bag.
    #[cold]
    #[inline(never)]
    pub(super) fn left_failed_landing(&self, id: NodeId, mark: u64, bag: &mut Deferred<C>) {
        defer(self.release_left(id, mark), bag);
    }

    fn release_left(&self, id: NodeId, mark: u64) -> Vec<Ticket> {
        let mut t = self.tickets.write();
        let stale = release(&mut t, mark, slot_landings(id.0));
        self.sync_held(&t);
        stale
    }

    /// Node `slot` at `generation` is being deleted under the runtime's
    /// lock: the tickets it alone kept wait for the delete's bag.
    #[inline(always)]
    pub(super) fn bury_tickets(&self, slot: u32, generation: u32) {
        if self.tickets_held() {
            self.bury(slot, generation);
        }
        if self.regens_held.load(Relaxed) {
            self.forget_failed_run(slot, generation);
        }
    }

    #[cold]
    #[inline(never)]
    fn bury(&self, slot: u32, generation: u32) {
        let mut t = self.tickets.write();
        let stale = release(&mut t, u64::MAX, (slot, generation)..=(slot, generation));
        t.buried.extend(stale);
    }

    /// A delete's bag drains: drops the tickets its deletes buried, and
    /// returns the panic of a payload's `Drop` for the bag to raise once
    /// its own items have dropped.
    #[inline(always)]
    pub(crate) fn drop_buried_tickets(&self) -> Option<Box<dyn Any + Send>> {
        if self.tickets_held() {
            self.drop_buried()
        } else {
            None
        }
    }

    #[cold]
    #[inline(never)]
    fn drop_buried(&self) -> Option<Box<dyn Any + Send>> {
        let buried = {
            let mut t = self.tickets.write();
            let buried = std::mem::take(&mut t.buried);
            self.sync_held(&t);
            buried
        };
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(buried))).err()
    }

    /// A compute's sweep released nodes: the tickets they buried join its
    /// bag, which drops them as it drains.
    #[inline(always)]
    pub(super) fn defer_buried_tickets(&self, bag: &mut Deferred<C>) {
        if self.tickets_held() {
            self.defer_buried(bag);
        }
    }

    #[cold]
    #[inline(never)]
    fn defer_buried(&self, bag: &mut Deferred<C>) {
        let buried = {
            let mut t = self.tickets.write();
            let buried = std::mem::take(&mut t.buried);
            self.sync_held(&t);
            buried
        };
        defer(buried, bag);
    }

    /// Notes `nodes`, landings a read took `payload` for, just above the
    /// reading frame, so that a walk's entry below that contains the panic
    /// again holds them too.
    fn note_read(&self, nodes: Vec<(u32, u32)>, payload: &(dyn Any + Send), restorable: bool) {
        self.note_above(nodes.into_iter().map(|n| (n, 0)), payload, restorable);
    }

    /// Notes each of `nodes` at its depth above the reading frame.
    fn note_above(
        &self,
        nodes: impl IntoIterator<Item = ((u32, u32), u32)>,
        payload: &(dyn Any + Send),
        restorable: bool,
    ) {
        let (rt, base, at) = (
            self.runtime_id,
            self.dep_stack.next_cidx(),
            payload_key(payload),
        );
        let mut nodes = nodes.into_iter().peekable();
        if nodes.peek().is_none() {
            return;
        }
        with_notes(|n| {
            for (node, depth) in nodes {
                n.push(rt, at, base + depth, node, restorable);
            }
        });
    }

    /// A delivered unwind a retry's read raises: its landings follow the
    /// copy the read takes, each as deep above the read as it landed, so
    /// the retry's later pops retire the deeper ones as the unbounded
    /// run's do.
    pub(super) fn note_delivered(&self, nodes: Landings, payload: &(dyn Any + Send)) {
        self.note_above(nodes.into_nodes(), payload, false);
    }

    /// A delivery keeps a copy of `payload` for a frame's retry while
    /// `payload` unwinds on: the copy carries the landings noted for it,
    /// with their depths above `base`, the position just above the
    /// reading frame. The retry's read stands for the frame's read in the
    /// unbounded run, so a discard of the run the payload unwinds through
    /// no longer holds their ticket again.
    pub(super) fn copy_notes(&self, payload: &(dyn Any + Send), base: u32) -> Landings {
        if !notes_held() {
            return Landings::default();
        }
        let (rt, at) = (self.runtime_id, payload_key(payload));
        let landings = with_notes(|n| n.copy(rt, at, base));
        replay::mark_copied(rt, &landings);
        landings
    }

    /// A delivery takes `payload` itself for a frame's retry, so its
    /// unwind ends here and the delivery carries its landings, with their
    /// depths above `base` as [`Self::copy_notes`] gives them.
    pub(super) fn take_notes(&self, payload: &(dyn Any + Send), base: u32) -> Landings {
        if !notes_held() {
            return Landings::default();
        }
        let (rt, at) = (self.runtime_id, payload_key(payload));
        let landings = with_notes(|n| n.take(rt, at, base));
        replay::mark_copied(rt, &landings);
        landings
    }

    /// Applies `ev` to the copies of this runtime's notes that kept reads
    /// and passed deliveries hold, as it applies to the notes: a frame's
    /// retry re-notes them, and the unbounded run's frame would find
    /// only those still noted. A speculative run is no event of the
    /// unbounded run's.
    pub(super) fn retire_copies(&self, ev: &Retire) {
        if !replay::speculative() && replay::copies_held() {
            replay::retire_copies(self.runtime_id, ev);
        }
    }

    /// An unwind stopped at frame `cidx`: a closure there caught it, or a
    /// walk's entry there contained it. The notes above are spent.
    #[cold]
    #[inline(never)]
    pub(super) fn retire_notes_above(&self, cidx: u32) {
        if notes_held() {
            with_notes(|n| n.forget_from(cidx + 1, None));
        }
    }

    /// An unwind reached an outermost entry, whose frames have popped.
    #[cold]
    #[inline(never)]
    pub(super) fn retire_notes(&self) {
        if notes_held() {
            let top = self.dep_stack.next_cidx();
            with_notes(|n| n.forget_from(top, None));
        }
    }

    /// A discarded run's `payload`, about to be dropped: the run never
    /// happened, so a ticket its reads took is held again, with the
    /// payload, for those of its landings no other landing has replaced.
    /// Returns the payload when no read took one.
    #[cold]
    #[inline(never)]
    pub(super) fn restore_contained(
        &self,
        payload: Box<dyn Any + Send>,
    ) -> Option<Box<dyn Any + Send>> {
        if !notes_held() {
            return Some(payload);
        }
        let (rt, at) = (self.runtime_id, payload_key(&*payload));
        let nodes = with_notes(|n| n.restore(rt, at));
        self.hold(nodes, payload)
    }

    /// The contained payloads still held, for tests.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn contained_payloads(&self) -> usize {
        let t = self.tickets.read();
        t.held.len() + t.buried.len()
    }
}
