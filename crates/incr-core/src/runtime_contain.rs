//! A walk's containment of what its cleaning raises. The dirty walk
//! cleans every recorded dep of a node before rerunning it, so it can
//! compute a dep the rerun no longer reads. A non-root walk entry whose
//! compute panics therefore lands as a changed dep and the walk goes on,
//! while a cycle goes on to the reader. A panic's payload stays with the
//! Failed landings it made, as a ticket, for the first read of any of
//! them; a closure that panics after catching a cycle lands unwound,
//! keeps no payload, and runs again for the next read.
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
use std::collections::{BTreeMap, BTreeSet};
use std::ops::RangeInclusive;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex, PoisonError};

use super::*;
use crate::locks::Lock;
use crate::notes::{removal, Landed, Members, Notes, Stamped};
use crate::replay::{note_work, payload_key, Landings, Retire};

/// A Failed read's node left Failed before the read could tell which
/// landing it saw.
pub(super) struct Left;

/// The tickets every thread's walks hold.
#[derive(Default)]
pub(crate) struct Tickets {
    next: u64,
    /// By id. A ticket keeps its id while reads take it and walks' entries
    /// hold it again, so that its landings move with it.
    held: BTreeMap<u64, Ticket>,
    /// The tickets reads took, with their payloads, until a walk's entry
    /// holds one again or the read's note goes.
    lent: BTreeMap<u64, Ticket>,
    /// The lent tickets whose read's note went, which `prune` lets go.
    dropped: Arc<Mutex<Vec<u64>>>,
    /// The ticket of each landing a held or lent ticket keeps, which is one
    /// at most, with the landing's place among its members and the order it
    /// joined in, so that a node's flip to Dirty removes it only from a
    /// ticket it joined before the flip. A landing whose ticket went is in
    /// none.
    landings: BTreeMap<(u32, u32), Joined>,
    /// The tickets of deleted nodes and of nodes a set's walk flipped,
    /// dropped once the delete, sweep or walk that released them holds no
    /// lock and has finished.
    buried: Vec<Ticket>,
    /// By the slot of a dep a set's walk stops at, the Failed landings over
    /// it (slot, generation) that no walk flagged; see `note_past_sets`.
    past: BTreeMap<u32, Vec<(u32, u32)>>,
    /// By the slot of a Dirty node below a noted dep, the slots of the
    /// nodes that read it, up which a take goes to the noted deps.
    above: BTreeMap<u32, Vec<u32>>,
    /// The nodes whose Dirty deps `above` holds, each scanned once.
    scanned: BTreeSet<u32>,
    /// By slot, the generation and stamp of the node's latest Failed
    /// landing under `Shared`. A note names the landing it was made for by
    /// its stamp, so a containment that comes after the node was flipped
    /// and landed Failed again, on another thread, holds nothing for it.
    landed: BTreeMap<u32, (u32, u64)>,
}

/// `slot`'s bit in `Runtime::failed_past`: a walk takes the tickets' lock
/// only at a dep whose bit is set; notes under other residues never set it.
#[inline(always)]
pub(super) fn past_bit(slot: u32) -> u64 {
    1 << (slot & 63)
}

#[derive(Clone, Copy)]
struct Joined {
    ticket: u64,
    member: u32,
    seq: u64,
}

/// A contained panic, held while any of its members, the `(node,
/// generation)` landings it failed at or above the entry's position, stays
/// in that Failed landing. A node's later Failed landing, its flip to Dirty
/// or its delete removes it.
struct Ticket {
    members: Arc<Members>,
    live: u32,
    /// The mutex only makes the table `Sync`: the payload moves in and out
    /// under the write lock. None while a read has it.
    payload: Option<Mutex<Box<dyn Any + Send>>>,
}

impl Tickets {
    /// The ticket landing `n` is in, if any, and whether it is held.
    fn of(&self, n: (u32, u32)) -> Option<(Joined, bool)> {
        let j = *self.landings.get(&n)?;
        if self.held.contains_key(&j.ticket) {
            Some((j, true))
        } else {
            self.lent.contains_key(&j.ticket).then_some((j, false))
        }
    }

    /// Takes `j`'s landing out of its ticket's members at removal `at`.
    fn leave(&mut self, j: Joined, at: u64) -> Option<&mut Ticket> {
        let c = match self.held.get_mut(&j.ticket) {
            Some(c) => c,
            None => self.lent.get_mut(&j.ticket)?,
        };
        c.members.lock().members[j.member as usize].gone = at;
        c.live -= 1;
        Some(c)
    }

    /// Lets the tickets go whose read's note went without holding them
    /// again, with their landings.
    fn prune(&mut self) {
        let dropped =
            std::mem::take(&mut *self.dropped.lock().unwrap_or_else(PoisonError::into_inner));
        note_work(dropped.len());
        for id in dropped {
            let Some(c) = self.lent.remove(&id) else {
                continue;
            };
            let members = c.members.lock();
            note_work(members.members.len());
            for m in &members.members {
                if self.landings.get(&m.node).is_some_and(|j| j.ticket == id) {
                    self.landings.remove(&m.node);
                }
            }
        }
    }
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
            if !holds
                && value_code::thread_word() & (value_code::KEPT_BIT | value_code::REPLAY_BIT) == 0
            {
                replay::end_kept_marks();
            }
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
    let found: Vec<((u32, u32), Joined)> = t
        .landings
        .range(landings)
        .filter(|&(_, j)| j.seq < mark)
        .map(|(&landing, &j)| (landing, j))
        .collect();
    let mut stale = Vec::new();
    note_work(found.len());
    let at = removal();
    for (landing, j) in found {
        t.landings.remove(&landing);
        // A lent ticket keeps its payload's read waiting to hold it again.
        if t.leave(j, at)
            .is_some_and(|c| c.live == 0 && c.payload.is_some())
        {
            stale.extend(t.held.remove(&j.ticket));
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
        self.tickets_held.store(
            !t.held.is_empty() || !t.lent.is_empty() || !t.buried.is_empty(),
            Relaxed,
        );
    }

    /// The payload a walk contained for `id`'s current Failed landing, taken
    /// by the first read of any node it failed.
    ///
    /// The ticket is noted against the payload just above the reading
    /// frame: a read from a compute whose panic a walk's entry below
    /// contains again is no reader's observation, so that entry holds the
    /// ticket again. A speculative run's read takes none: a closure in
    /// the run can catch and drop the payload before the run's discard
    /// would hold it again, and the real run's read is owed it.
    /// With tickets held, `Left` too when the slot no longer holds
    /// `generation`'s node; with none, the read's panic is the one it would
    /// have made before the delete.
    pub(super) fn take_contained(
        &self,
        id: NodeId,
        generation: Option<u32>,
    ) -> Result<Option<Box<dyn Any + Send>>, Left> {
        if !self.tickets_held() || replay::speculative() {
            return Ok(None);
        }
        let node = self.nodes.get(id.0);
        let mut t = self.tickets.write();
        // Every Failed landing removes the node from the tickets before its
        // store, so under the lock a node still Failed at this generation
        // is in the latest landing's ticket, if any.
        if self.seen(node.state_cell(), state::load::<C>(node.state_cell())) != NodeState::Failed {
            return Err(Left);
        }
        // After the state, as a recycle bumps it before its new node lands.
        let landing = (id.0, node.generation());
        if generation.is_some_and(|g| g != landing.1) {
            return Err(Left);
        }
        t.prune();
        let Some((j, true)) = t.of(landing) else {
            return Ok(None);
        };
        let mut c = t
            .held
            .remove(&j.ticket)
            .expect("incr-core: a held ticket is listed");
        let payload = c
            .payload
            .take()
            .expect("incr-core: a held ticket has its payload");
        let base = self.dep_stack.next_cidx();
        // With no frame on the thread, no walk's entry can hold the ticket
        // again, so it goes as its members are noted, rather than lent.
        let alone = (base == 0).then(|| {
            let live: Vec<Stamped> = c
                .members
                .lock()
                .members
                .iter()
                .filter(|m| m.gone == u64::MAX)
                .map(|m| (m.node, m.at))
                .collect();
            note_work(live.len());
            for (n, _) in &live {
                t.landings.remove(n);
            }
            live
        });
        let members = c.members.clone();
        if alone.is_none() {
            t.lent.insert(j.ticket, c);
        }
        self.sync_held(&t);
        drop(t);
        let payload = payload.into_inner().unwrap_or_else(PoisonError::into_inner);
        match alone {
            Some(nodes) => {
                self.note_above(nodes.into_iter().map(|(n, at)| (n, 0, at)), &*payload, true)
            }
            None => {
                let (rt, at) = (self.runtime_id, payload_key(&*payload));
                with_notes(|n| n.push_lent(rt, at, base, members));
            }
        }
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
        let stamp = if C::SINGLE_THREADED {
            0
        } else {
            let mut t = self.tickets.write();
            let stamp = t.next;
            t.next += 1;
            t.landed.insert(id.0, (generation, stamp));
            stamp
        };
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
                n.push(rt, at, cidx, (landing, stamp), false);
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
        self.park(ParkPoint::BeforeContain);
        let (rt, at) = (self.runtime_id, payload_key(&*payload));
        self.retire_copies(&Retire::Contained { cidx, payload: at });
        let (nodes, lent) = if notes_held() {
            with_notes(|n| n.contain(rt, cidx, at))
        } else {
            (Vec::new(), Vec::new())
        };
        if let Some(payload) = self.hold(nodes, lent, payload) {
            bag.push_erased(payload);
        }
    }

    /// Holds `payload` for the tickets in `lent` that reads took with it,
    /// as one, and for those of `nodes` still in their Failed landing and
    /// in no held ticket, or returns it when none is. A node in a ticket a
    /// read took leaves that ticket.
    fn hold(
        &self,
        mut nodes: Vec<Stamped>,
        lent: Vec<Arc<Members>>,
        payload: Box<dyn Any + Send>,
    ) -> Option<Box<dyn Any + Send>> {
        if nodes.is_empty() && lent.is_empty() {
            return Some(payload);
        }
        let mut t = self.tickets.write();
        t.prune();
        let mut taken: Vec<(u64, Ticket)> = lent
            .iter()
            .filter_map(|m| t.lent.remove(&m.id).map(|c| (m.id, c)))
            .collect();
        note_work(nodes.len() + taken.len());
        nodes.retain(|&((slot, generation), stamp)| {
            let node = self.nodes.get(slot);
            node.generation() == generation
                && (C::SINGLE_THREADED || t.landed.get(&slot) == Some(&(generation, stamp)))
                && self.in_failed_landing(
                    slot,
                    self.seen_quietly(node.state_cell(), state::load::<C>(node.state_cell())),
                )
                && t.of((slot, generation))
                    .is_none_or(|(j, held)| !held && !taken.iter().any(|&(id, _)| id == j.ticket))
        });
        let seq = t.next;
        t.next += 1;
        let (id, mut ticket) = taken.pop().unwrap_or_else(|| {
            (
                seq,
                Ticket {
                    members: Arc::new(Members::new(seq, t.dropped.clone())),
                    live: 0,
                    payload: None,
                },
            )
        });
        let at = removal();
        // Tickets reads took with the same payload go into one.
        for (other, c) in taken {
            let members = c.members.lock();
            note_work(members.members.len());
            for m in members.members.iter().filter(|m| m.gone == u64::MAX) {
                let j = t.landings[&m.node];
                debug_assert_eq!(j.ticket, other);
                nodes.push((m.node, m.at));
                t.landings.remove(&m.node);
            }
        }
        for (n, stamp) in nodes {
            if let Some((j, _)) = t.of(n) {
                t.leave(j, at);
            }
            let member = {
                let mut joined = ticket.members.lock();
                joined.members.push(crate::notes::Member {
                    node: n,
                    at: stamp,
                    gone: u64::MAX,
                });
                joined.members.len() as u32 - 1
            };
            t.landings.insert(
                n,
                Joined {
                    ticket: id,
                    member,
                    seq,
                },
            );
            ticket.live += 1;
        }
        if cfg!(test) {
            self.check_members(&t, id, &ticket);
        }
        if ticket.live == 0 {
            self.sync_held(&t);
            return Some(payload);
        }
        ticket.payload = Some(Mutex::new(payload));
        t.held.insert(id, ticket);
        self.sync_held(&t);
        None
    }

    /// Whether `slot`'s node, in `state` as this thread sees it, is in its
    /// Failed landing: Failed, or Computing under a regenerating run, which
    /// lands nothing and puts it back. Exact under the tickets lock, under
    /// which a rerun moves it and marks it.
    fn in_failed_landing(&self, slot: u32, state: NodeState) -> bool {
        match state {
            NodeState::Failed => true,
            NodeState::Computing | NodeState::ComputingDirty => {
                crate::replay::is_regen_running(self.runtime_id, slot)
            }
            _ => false,
        }
    }

    /// [`Self::in_failed_landing`] for `slot`'s node of `generation`. The
    /// write hold excludes a rerun, which moves the node and marks it under a
    /// read hold.
    pub(super) fn failed_landing_of(&self, slot: u32, generation: u32) -> bool {
        let _t = self.tickets.write();
        let node = self.nodes.get(slot);
        node.generation() == generation
            && self.in_failed_landing(
                slot,
                self.seen_quietly(node.state_cell(), state::load::<C>(node.state_cell())),
            )
    }

    /// Every landing a ticket keeps is in its Failed landing and indexed to
    /// it: every change from that landing releases it.
    fn check_members(&self, t: &Tickets, id: u64, ticket: &Ticket) {
        let members = ticket.members.lock();
        let live = members.members.iter().filter(|m| m.gone == u64::MAX);
        let mut n = 0;
        for m in live {
            n += 1;
            let node = self.nodes.get(m.node.0);
            // Another thread's held landing shows Computing here until its
            // publication, which releases what it does not land Failed.
            assert!(
                node.generation() == m.node.1
                    && (node.held_mark()
                        || self.in_failed_landing(
                            m.node.0,
                            self.seen_quietly(
                                node.state_cell(),
                                state::load::<C>(node.state_cell()),
                            ),
                        ))
                    && t.landings.get(&m.node).is_some_and(|j| j.ticket == id),
                "incr-core: a ticket keeps only Failed landings it indexes"
            );
        }
        assert_eq!(n, ticket.live, "incr-core: a ticket counts its landings");
    }

    /// A set's walk flips `id` from Failed to Dirty, under the tickets'
    /// lock: a `hold` of the landing either precedes the flip, which then
    /// releases its ticket, or sees the node Dirty, or Failed in a later
    /// landing with another stamp (`Tickets::landed`), and holds nothing. The
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
        self.park(ParkPoint::FlipBeforeCas(id));
        let mut t = self.tickets.write();
        let flipped = self.with_failed_run_token(id.0, node.generation(), |token| {
            state::try_transition::<C>(node.state_cell(), from, NodeState::Dirty)
                .is_ok()
                .then_some(token)
        });
        let Some(token) = flipped else {
            return false;
        };
        if from != NodeState::Failed {
            // With the flip, under the lock: a compute may take the node
            // from Dirty once the lock goes.
            crate::replay::regen_running(self.runtime_id, id.0, false);
        }
        let stale = release(&mut t, u64::MAX, slot_landings(id.0));
        t.buried.extend(stale);
        self.sync_held(&t);
        drop(t);
        self.park(ParkPoint::FlippedFailed(id));
        self.forget_failed_run_if(id.0, node.generation(), token);
        true
    }

    /// A retraction lands the held Failed landings of `slots` unrun: their
    /// tickets let go of them as a flip's do.
    #[cold]
    #[inline(never)]
    pub(super) fn release_retracted(&self, slots: &[u32], bag: &mut Deferred<C>) {
        if slots.is_empty() || !self.tickets_held() {
            return;
        }
        let mut t = self.tickets.write();
        let mut stale = Vec::new();
        for &slot in slots {
            stale.extend(release(&mut t, u64::MAX, slot_landings(slot)));
        }
        self.sync_held(&t);
        drop(t);
        defer(stale, bag);
    }

    /// Before `id` lands Failed over `deps`. A set's walk stops at a Dirty
    /// dep, whose dependents the walk that made it Dirty flagged; a dep a
    /// caught panic left Dirty had no such walk. The landing is noted under
    /// each such dep for the next walk that reaches it or a Dirty node below
    /// it, where a walk from further down stops first (0x5c05), so it lasts
    /// its revision and not past the dep's next change, except under
    /// `Shared` for a set whose walk passed the dep before the note was
    /// made. A dep is judged as this thread sees it, so a landing it holds
    /// toward Dirty is noted, and so is one a set flagged while held, which
    /// publishes as flagged, not as its target. A dep still Computing is
    /// not noted: on
    /// this thread it is an open cycle head the landing belongs to, and a
    /// flip there reruns the caught cycle on every read (0x1daa).
    /// Speculative landings are noted too: the take checks the node.
    #[inline(always)]
    pub(super) fn note_past_sets(&self, id: NodeId, generation: u32, deps: &[(NodeId, u32)]) {
        let past = |dep: NodeId| self.left_past(dep);
        if deps.iter().any(|&(d, _)| past(d)) {
            self.note_past(id, generation, deps, past);
        }
    }

    /// A held Failed landing's new deps published Failed: their reverse
    /// edges wait for its publication, so a set's flip of one reaches the
    /// landing only through a note. Taken before the publish reads whether
    /// they are settled: a flip before the note leaves the dep Dirty for
    /// that read, and one after finds the note. A dep this thread holds
    /// needs none, as no other thread reruns it before it publishes with
    /// the landing, after every held edge is in.
    #[cold]
    #[inline(never)]
    pub(super) fn note_held_failed_deps(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        deps: &[(NodeId, u32)],
    ) {
        let failed: Vec<&(NodeId, u32)> = deps
            .iter()
            .filter(|&&(d, _)| self.nodes.get(d.0).state() == NodeState::Failed)
            .collect();
        if failed.is_empty() {
            return;
        }
        let mut published: Vec<NodeId> = Vec::new();
        node.for_each_dep(|d| published.push(d));
        fn unpublished<'a, S: DepLookup + ?Sized>(
            failed: Vec<&'a (NodeId, u32)>,
            published: &S,
        ) -> Vec<&'a (NodeId, u32)> {
            failed
                .into_iter()
                .filter(|(d, _)| !published.has(*d))
                .collect()
        }
        let new = if dep_set::needs_hashing(published.len(), failed.len()) {
            unpublished(failed, &DepHashSet::new(&published))
        } else {
            unpublished(failed, published.as_slice())
        };
        if !new.is_empty() {
            self.note_held(id, node.generation(), new.into_iter());
        }
    }

    /// Notes `id` under `deps` as the reverse edges its publication installs
    /// would reach it: only a walk that flips a dep takes it, and a walk
    /// stopping below a dep reaches it through the dep's own notes, so no
    /// scan below, which would flip it where the published landing stays.
    fn note_held<'a>(
        &self,
        id: NodeId,
        generation: u32,
        deps: impl Iterator<Item = &'a (NodeId, u32)>,
    ) {
        let mut t = self.tickets.write();
        for &(dep, _) in deps {
            let notes = t.past.entry(dep.0).or_default();
            if !notes.contains(&(id.0, generation)) {
                notes.push((id.0, generation));
            }
            self.failed_past.fetch_or(past_bit(dep.0), Relaxed);
        }
    }

    #[inline(always)]
    fn left_past(&self, dep: NodeId) -> bool {
        let cell = self.nodes.get(dep.0).state_cell();
        let raw = state::load::<C>(cell);
        raw == NodeState::ComputingDirty
            || matches!(
                self.seen_quietly(cell, raw),
                NodeState::Dirty | NodeState::ComputingDirty
            )
    }

    #[cold]
    #[inline(never)]
    fn note_past(
        &self,
        id: NodeId,
        generation: u32,
        deps: &[(NodeId, u32)],
        past: impl Fn(NodeId) -> bool,
    ) {
        let mut t = self.tickets.write();
        for &(dep, _) in deps.iter().filter(|&&(d, _)| past(d)) {
            #[cfg(test)]
            self.past_note_work.fetch_add(1, Relaxed);
            let notes = t.past.entry(dep.0).or_default();
            if !notes.contains(&(id.0, generation)) {
                notes.push((id.0, generation));
            }
            self.failed_past.fetch_or(past_bit(dep.0), Relaxed);
            self.scan_below(&mut t, dep);
        }
    }

    /// Records, once per node until a take passes it, the deps below `dep`
    /// a walk may stop at with the nodes that read them. A node scanned
    /// before keeps the deps it had: its deps change only by a run, after a
    /// set below it that a take on its way up ended.
    fn scan_below(&self, t: &mut Tickets, dep: NodeId) {
        let mut stack = vec![dep];
        while let Some(node) = stack.pop() {
            if !t.scanned.insert(node.0) {
                continue;
            }
            self.nodes.get(node.0).for_each_dep(|below| {
                if self.stops_walks(below) {
                    #[cfg(test)]
                    self.past_note_work.fetch_add(1, Relaxed);
                    let above = t.above.entry(below.0).or_default();
                    if !above.contains(&node.0) {
                        above.push(node.0);
                    }
                    self.failed_past.fetch_or(past_bit(below.0), Relaxed);
                    stack.push(below);
                }
            });
        }
    }

    /// `id` installed a new dep list: if it was scanned, its Dirty deps are
    /// scanned again, as `above` may lack its new ones and the notes above
    /// it stand (a closure that reads by run count reruns a memo with no
    /// set and no take).
    #[inline(always)]
    pub(super) fn deps_installed(&self, id: u32) {
        if self.failed_past.load(Relaxed) & past_bit(id) != 0 {
            self.unscan(id);
        }
    }

    #[cold]
    #[inline(never)]
    fn unscan(&self, id: u32) {
        let mut t = self.tickets.write();
        if t.scanned.remove(&id) {
            self.scan_below(&mut t, NodeId(id));
        }
    }

    /// A node below a noted dep that a later walk may stop at: Dirty, or
    /// still Computing, which a budget's unwind lands Dirty before the
    /// unbounded run's landing does, so the scans of both agree (0x9593).
    fn stops_walks(&self, dep: NodeId) -> bool {
        state::load::<C>(self.nodes.get(dep.0).state_cell()) == NodeState::Computing
            || self.left_past(dep)
    }

    /// Recounts the bits only while few slots are noted, so a take stays
    /// within the entries it removes; past that the bits stay a superset.
    fn sync_past(&self, t: &Tickets) {
        if t.past.len() + t.above.len() > 64 {
            return;
        }
        let bits =
            (t.past.keys().chain(t.above.keys())).fold(0, |bits, &slot| bits | past_bit(slot));
        self.failed_past.store(bits, Relaxed);
    }

    /// A set's walk reached `dep`: queues the landings noted over it that
    /// are still Failed, or still Computing toward a Failed landing, which
    /// the walk then flags so that it lands Dirty.
    #[cold]
    #[inline(never)]
    pub(super) fn take_past_sets(&self, dep: NodeId, queue: &mut Vec<NodeId>) {
        let noted = |t: &Tickets| t.past.contains_key(&dep.0) || t.above.contains_key(&dep.0);
        if !noted(&self.tickets.read()) {
            return;
        }
        let mut t = self.tickets.write();
        if !noted(&t) {
            return;
        }
        let mut notes = Vec::new();
        let mut up = vec![dep.0];
        while let Some(slot) = up.pop() {
            t.scanned.remove(&slot);
            notes.extend(t.past.remove(&slot).into_iter().flatten());
            up.extend(t.above.remove(&slot).into_iter().flatten());
        }
        self.sync_past(&t);
        drop(t);
        for (id, generation) in notes {
            let node = self.nodes.get(id);
            if node.generation() == generation
                && matches!(
                    state::load::<C>(node.state_cell()),
                    NodeState::Failed | NodeState::Computing | NodeState::ComputingDirty
                )
            {
                queue.push(NodeId(id));
            }
        }
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

    /// A mark for [`Self::left_failed_landing`] that leaves the tickets
    /// joined from here on, which a later landing's readers took.
    pub(super) fn ticket_mark(&self) -> u64 {
        if !self.tickets_held() {
            return 0;
        }
        self.tickets.read().next
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
        if self.failed_past.load(Relaxed) & past_bit(slot) != 0 {
            self.bury_past(slot);
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

    #[cold]
    #[inline(never)]
    fn bury_past(&self, slot: u32) {
        let mut t = self.tickets.write();
        t.scanned.remove(&slot);
        if t.past.remove(&slot).is_some() | t.above.remove(&slot).is_some() {
            self.sync_past(&t);
        }
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
        let _drops = (!C::SINGLE_THREADED).then(|| crate::attempt::Outside::enter(true));
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

    /// Notes each of `nodes` at its depth above the reading frame.
    fn note_above(
        &self,
        nodes: impl IntoIterator<Item = Landed>,
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
            for (node, depth, stamp) in nodes {
                n.push(rt, at, base + depth, (node, stamp), restorable);
            }
        });
    }

    /// A delivered unwind a retry's read raises: its landings follow the
    /// copy the read takes, each as deep above the read as it landed, so
    /// the retry's later pops retire the deeper ones as the unbounded
    /// run's do.
    pub(super) fn note_delivered(&self, nodes: Landings, payload: &(dyn Any + Send)) {
        let Some(copy) = nodes.as_shared() else {
            self.note_above(nodes.into_nodes(), payload, false);
            return;
        };
        let (rt, base, at) = (
            self.runtime_id,
            self.dep_stack.next_cidx(),
            payload_key(payload),
        );
        with_notes(|n| n.note_copy(rt, at, base, copy));
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
        let (nodes, lent) = with_notes(|n| n.restore(rt, at));
        self.hold(nodes, lent, payload)
    }

    /// The tickets held, lent to reads or buried, and whether the runtime
    /// says any is, for tests.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn tickets_kept(&self) -> (usize, bool) {
        let t = self.tickets.read();
        (
            t.held.len() + t.lent.len() + t.buried.len(),
            self.tickets_held(),
        )
    }

    /// The contained payloads still held, for tests.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn contained_payloads(&self) -> usize {
        let t = self.tickets.read();
        t.held.len() + t.buried.len()
    }
}
