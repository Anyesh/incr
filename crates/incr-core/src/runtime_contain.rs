//! A walk's containment of what its cleaning raises. The dirty walk
//! cleans every recorded dep of a node before rerunning it, so it can
//! compute a dep the rerun no longer reads. A non-root walk entry whose
//! compute panics or closes a cycle therefore lands as a changed dep and
//! the walk goes on; a panic's payload stays with the Failed landings it
//! made, as a ticket, for the first read of any of them.
//!
//! The landings an unwind makes above the containing entry are found
//! through notes this thread keeps per landing, keyed by the address of
//! the panic's box. A note lives only while that box unwinds: every point
//! an unwind stops retires the notes above it, and a box the engine moves
//! into a delivery or drops takes its notes with it, so no later box at a
//! reused address meets them.

use std::any::Any;
use std::cell::RefCell;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Mutex, PoisonError};

use super::*;
use crate::locks::Lock;

/// A Failed read's node left Failed before the read could tell which
/// landing it saw.
pub(super) struct Left;

/// The tickets every thread's walks hold.
#[derive(Default)]
pub(crate) struct Tickets {
    next: u64,
    held: Vec<Ticket>,
    /// The tickets of deleted nodes, dropped by the delete's bag once it
    /// holds no lock.
    buried: Vec<Ticket>,
}

/// A contained panic, held while any of `nodes`, the `(node, generation)`
/// landings it failed, stays in that Failed landing: the entry's and each
/// one the unwind made above the entry's frame. A node's later Failed
/// landing, its flip to Dirty or its delete removes it.
struct Ticket {
    /// Orders tickets, so that a node's flip to Dirty removes it only from
    /// tickets of landings before the flip.
    seq: u64,
    nodes: Vec<(u32, u32)>,
    /// The mutex only makes the table `Sync`: the payload moves in and out
    /// under the write lock.
    payload: Mutex<Box<dyn Any + Send>>,
}

pub(crate) fn tickets<C: Cells>() -> C::Lock<Tickets> {
    <C::Lock<Tickets> as Lock<Tickets>>::new(Tickets::default())
}

/// A Failed landing a panic in flight on this thread made, or one of the
/// landings a read raised again from a ticket or a delivery.
struct Unwound {
    rt: RuntimeId,
    /// The frame position of the landing, on the thread's frame stack,
    /// which every runtime on the thread shares.
    cidx: u32,
    node: (u32, u32),
    payload: usize,
    /// Taken from a ticket by a read: a discarded run's read was no
    /// reader's observation, so its drop holds the ticket again.
    restorable: bool,
}

#[cfg(not(loom))]
thread_local! {
    static UNWOUND: RefCell<Vec<Unwound>> = const { RefCell::new(Vec::new()) };
}

#[cfg(loom)]
loom::thread_local! {
    static UNWOUND: RefCell<Vec<Unwound>> = RefCell::new(Vec::new());
}

fn address(payload: &(dyn Any + Send)) -> usize {
    std::ptr::from_ref(payload).cast::<()>() as usize
}

/// Edits this thread's notes and keeps `value_code::NOTES_BIT` in step.
fn with_notes<R>(f: impl FnOnce(&mut Vec<Unwound>) -> R) -> R {
    UNWOUND.with(|u| {
        let mut u = u.borrow_mut();
        let held = !u.is_empty();
        let r = f(&mut u);
        let holds = !u.is_empty();
        if held != holds {
            value_code::set_notes_bit(holds);
        }
        r
    })
}

fn notes_held() -> bool {
    value_code::thread_word() & value_code::NOTES_BIT != 0
}

/// Removes `landing` from the tickets before `mark` (all of them for
/// `u64::MAX`), returning the payloads left with no landing, which the
/// caller drops outside the lock: a drop is user code.
fn release(t: &mut Tickets, mark: u64, landing: impl Fn((u32, u32)) -> bool) -> Vec<Ticket> {
    let mut stale = Vec::new();
    let mut i = 0;
    while i < t.held.len() {
        let c = &mut t.held[i];
        if c.seq < mark {
            c.nodes.retain(|&n| !landing(n));
        }
        if c.nodes.is_empty() {
            stale.push(t.held.swap_remove(i));
        } else {
            i += 1;
        }
    }
    stale
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
    /// holds them too.
    pub(super) fn take_contained(&self, id: NodeId) -> Result<Option<Box<dyn Any + Send>>, Left> {
        if !self.tickets_held() {
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
        let Some(at) = t.held.iter().position(|c| c.nodes.contains(&landing)) else {
            return Ok(None);
        };
        let Ticket { nodes, payload, .. } = t.held.swap_remove(at);
        self.sync_held(&t);
        drop(t);
        let payload = payload.into_inner().unwrap_or_else(PoisonError::into_inner);
        self.note_read(nodes, &*payload, true);
        Ok(Some(payload))
    }

    /// Before `id` lands Failed at frame `cidx`, from `land_failed`: drops
    /// `id` from the tickets of earlier landings, then holds `contained`,
    /// the panic a walk's entry contained, for this landing and those the
    /// unwind made above it, or notes `payload`, a panic unwinding on, for
    /// a walk's entry below to contain. Returns the mark that orders this
    /// landing after every ticket it removed from, or 0 with none held.
    #[cold]
    #[inline(never)]
    pub(super) fn failed_landing(
        &self,
        id: NodeId,
        generation: u32,
        cidx: u32,
        payload: Option<&(dyn Any + Send)>,
        contained: Option<Box<dyn Any + Send>>,
    ) -> u64 {
        let rt = self.runtime_id;
        let landing = (id.0, generation);
        let nodes = match (&contained, payload) {
            (Some(contained), _) => {
                let at = address(&**contained);
                let mut nodes = vec![landing];
                with_notes(|u| {
                    u.retain(|w| {
                        let above = w.cidx > cidx;
                        if above && w.rt == rt && w.payload == at && !nodes.contains(&w.node) {
                            nodes.push(w.node);
                        }
                        !above
                    });
                });
                nodes
            }
            (None, Some(payload)) => {
                // The frames at and above `cidx` belong to this unwind, so
                // an earlier unwind's notes there are spent.
                let at = address(payload);
                with_notes(|u| {
                    u.retain(|w| !(w.cidx >= cidx && w.payload != at));
                    u.push(Unwound {
                        rt,
                        cidx,
                        node: landing,
                        payload: at,
                        restorable: false,
                    });
                });
                Vec::new()
            }
            (None, None) => Vec::new(),
        };
        if contained.is_none() && !self.tickets_held() {
            return 0;
        }
        let (stale, mark) = {
            let mut t = self.tickets.write();
            let stale = release(&mut t, u64::MAX, |n| n == landing);
            if let Some(payload) = contained {
                let seq = t.next;
                t.held.push(Ticket {
                    seq,
                    nodes,
                    payload: Mutex::new(payload),
                });
            }
            t.next += 1;
            self.sync_held(&t);
            (stale, t.next)
        };
        drop(stale);
        mark
    }

    /// Where `id` is about to leave Failed for Dirty: the mark its
    /// `left_failed` takes, so that it keeps the ticket of a landing made
    /// after the flip. `None` when no ticket is held.
    #[cold]
    #[inline(never)]
    pub(super) fn ticket_mark(&self) -> Option<u64> {
        self.tickets_held().then(|| self.tickets.read().next)
    }

    /// `id` left the Failed landing that ordered before `mark`, so no read
    /// takes a payload for it any more.
    #[cold]
    #[inline(never)]
    pub(super) fn left_failed(&self, id: NodeId, mark: u64) {
        let stale = {
            let mut t = self.tickets.write();
            let stale = release(&mut t, mark, |(n, _)| n == id.0);
            self.sync_held(&t);
            stale
        };
        drop(stale);
    }

    /// Node `slot` at `generation` is being deleted under the runtime's
    /// lock: the tickets it alone kept wait for the delete's bag.
    #[inline(always)]
    pub(super) fn bury_tickets(&self, slot: u32, generation: u32) {
        if self.tickets_held() {
            self.bury(slot, generation);
        }
    }

    #[cold]
    #[inline(never)]
    fn bury(&self, slot: u32, generation: u32) {
        let mut t = self.tickets.write();
        let stale = release(&mut t, u64::MAX, |n| n == (slot, generation));
        t.buried.extend(stale);
    }

    /// A delete's bag drains: drops the tickets its deletes buried.
    #[inline(always)]
    pub(crate) fn drop_buried_tickets(&self) {
        if self.tickets_held() {
            self.drop_buried();
        }
    }

    #[cold]
    #[inline(never)]
    fn drop_buried(&self) {
        let buried = {
            let mut t = self.tickets.write();
            let buried = std::mem::take(&mut t.buried);
            self.sync_held(&t);
            buried
        };
        drop(buried);
    }

    /// Notes `nodes`, landings a read took `payload` for, just above the
    /// reading frame, so that a walk's entry below that contains the panic
    /// again holds them too.
    fn note_read(&self, nodes: Vec<(u32, u32)>, payload: &(dyn Any + Send), restorable: bool) {
        if nodes.is_empty() {
            return;
        }
        let (rt, cidx, at) = (
            self.runtime_id,
            self.dep_stack.next_cidx(),
            address(payload),
        );
        with_notes(|u| {
            u.extend(nodes.into_iter().map(|node| Unwound {
                rt,
                cidx,
                node,
                payload: at,
                restorable,
            }));
        });
    }

    /// A delivered unwind a retry's read raises: its landings follow the
    /// copy the read takes.
    pub(super) fn note_delivered(&self, nodes: Vec<(u32, u32)>, payload: &(dyn Any + Send)) {
        self.note_read(nodes, payload, false);
    }

    /// A delivery keeps a copy of `payload` for a frame's retry while
    /// `payload` unwinds on: the copy carries the landings noted for it.
    /// The retry's read stands for the frame's read in the unbounded run,
    /// so a discard of the run the payload unwinds through no longer holds
    /// their ticket again.
    pub(super) fn copy_notes(&self, payload: &(dyn Any + Send)) -> Vec<(u32, u32)> {
        if !notes_held() {
            return Vec::new();
        }
        let (rt, at) = (self.runtime_id, address(payload));
        let mut nodes = Vec::new();
        with_notes(|u| {
            for w in u.iter_mut().filter(|w| w.payload == at) {
                w.restorable = false;
                if w.rt == rt && !nodes.contains(&w.node) {
                    nodes.push(w.node);
                }
            }
        });
        nodes
    }

    /// A delivery takes `payload` itself for a frame's retry, so its
    /// unwind ends here and the delivery carries its landings.
    pub(super) fn take_notes(&self, payload: &(dyn Any + Send)) -> Vec<(u32, u32)> {
        if !notes_held() {
            return Vec::new();
        }
        let (rt, at) = (self.runtime_id, address(payload));
        let mut nodes = Vec::new();
        with_notes(|u| {
            u.retain(|w| {
                let same = w.payload == at;
                if same && w.rt == rt && !nodes.contains(&w.node) {
                    nodes.push(w.node);
                }
                !same
            });
        });
        nodes
    }

    /// An unwind stopped at frame `cidx`: a closure there caught it, or a
    /// walk's entry there contained it. The notes above are spent.
    #[cold]
    #[inline(never)]
    pub(super) fn retire_notes_above(&self, cidx: u32) {
        if notes_held() {
            with_notes(|u| u.retain(|w| w.cidx <= cidx));
        }
    }

    /// An unwind reached an outermost entry, whose frames have popped.
    #[cold]
    #[inline(never)]
    pub(super) fn retire_notes(&self) {
        if notes_held() {
            let top = self.dep_stack.next_cidx();
            with_notes(|u| u.retain(|w| w.cidx < top));
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
        let (rt, at) = (self.runtime_id, address(&*payload));
        let mut nodes = Vec::new();
        with_notes(|u| {
            u.retain(|w| {
                let same = w.payload == at;
                if same && w.rt == rt && w.restorable && !nodes.contains(&w.node) {
                    nodes.push(w.node);
                }
                !same
            });
        });
        if nodes.is_empty() {
            return Some(payload);
        }
        let mut t = self.tickets.write();
        nodes.retain(|&(slot, generation)| {
            let node = self.nodes.get(slot);
            node.generation() == generation
                && state::load::<C>(node.state_cell()) == NodeState::Failed
                && !t.held.iter().any(|c| c.nodes.contains(&(slot, generation)))
        });
        if nodes.is_empty() {
            return Some(payload);
        }
        let seq = t.next;
        t.next += 1;
        t.held.push(Ticket {
            seq,
            nodes,
            payload: Mutex::new(payload),
        });
        self.sync_held(&t);
        None
    }

    /// The contained payloads still held, for tests.
    #[cfg(all(test, not(loom)))]
    pub(crate) fn contained_payloads(&self) -> usize {
        self.tickets.read().held.len()
    }
}
