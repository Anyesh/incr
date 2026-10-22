//! Provenance of a user panic held for a virtual frame. This thread's
//! entry for Y, below which X's virtual frame reads Y, lands Y Failed and
//! holds the payload for X's retry; virtual frames hold no claim, so
//! another thread can meanwhile compute X, read Y Failed and land X
//! Failed. Only a landing that let that failed read's panic out unchanged
//! allows the serial order with this thread first, where the retry gets
//! the payload; any other allows only the other thread first, where the
//! retry reads X Failed. The record published before Y lands carries
//! which landings did, and keeps them: once a landing let the failure out,
//! the other thread has observed it, so the retry is owed the payload
//! whatever happens to the node after, a reland Clean or Failed or the
//! recycling of its slot. The replay driver's compute of a node checks for
//! it after each load of the node's state and again after its claim, which
//! can succeed on a Dirty that such a landing and a set made since. A
//! Failed read also takes, with no record, the payload held for its
//! node's own frame along the read's path once this thread stored a
//! Failed landing of the node that the same read's payload left
//! (`path_delivery`), under `Local` and `Shared` alike.

use std::any::Any;
use std::panic::resume_unwind;
use std::rc::Rc;
use std::sync::Arc;

use super::*;
use crate::locks::Lock;
use crate::replay::{self, Carry, Provenance, ReadKey};

/// The read a record is for: the virtual frame's node and generation,
/// the node it was blocked reading and which of its reads of it that was.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Reader {
    node: u32,
    generation: u32,
    reading: u64,
    ordinal: u32,
}

impl From<ReadKey> for Reader {
    fn from(k: ReadKey) -> Self {
        Self {
            node: k.node,
            generation: k.generation,
            reading: k.reading,
            ordinal: k.ordinal,
        }
    }
}

struct Record {
    id: u64,
    reader: Reader,
    /// The nodes, as `(node, generation)`, a Failed landing of which on
    /// another thread let the record's read failure out unchanged. A later
    /// landing never removes one.
    propagated: Vec<(u32, u32)>,
}

/// The records of the payloads held on every thread, each live while a
/// delivery holds it.
#[derive(Default)]
pub(crate) struct Records {
    next: u64,
    live: Vec<Record>,
}

pub(crate) type Table<C> = Arc<<C as Cells>::Lock<Records>>;

pub(crate) fn table<C: Cells>() -> Table<C> {
    Arc::new(<C::Lock<Records> as Lock<Records>>::new(Records::default()))
}

/// A delivery's hold on its record, which the last hold's drop removes.
struct Hold<C: Cells> {
    table: Table<C>,
    id: u64,
}

impl<C: Cells> Provenance for Hold<C> {
    fn record(&self) -> u64 {
        self.id
    }
}

impl<C: Cells> Drop for Hold<C> {
    fn drop(&mut self) {
        self.table.write().live.retain(|r| r.id != self.id);
    }
}

/// The message of a read of node `slot` landed Failed.
fn failed_read_message(slot: u32) -> String {
    format!(
        "incr-core: node {slot} read but its last compute panicked; it stays Failed until a \
         dependency changes (set an upstream input to retry)"
    )
}

impl<C: Cells> Runtime<C> {
    /// `compute_attempts`' Failed arm: the read panics "last compute
    /// panicked", or takes the payload a walk contained for the landing it
    /// sees (`take_contained`), or else the one held for `id`'s frame
    /// along the read's path (`path_delivery`). Returns
    /// when the node has left Failed since the read saw it, to be read
    /// again. A payload a record owes the read was taken after the load,
    /// by `owed_payload`.
    #[cold]
    #[inline(never)]
    pub(super) fn read_failed_node<const MODE: u8>(&self, id: NodeId, bag: &mut Deferred<C>) {
        let Ok(payload) = self.take_contained(id) else {
            return;
        };
        if MODE & VIRT != 0 {
            let generation = self.nodes.get(id.0).generation();
            if let Some((context, i)) = self.path_delivery(id, generation) {
                // A ticket's landing is newer than the delivery's round, so
                // the unbounded run's read gets the ticket's payload and
                // the frame's run the delivery stands for never happens.
                if payload.is_none() {
                    let payload = self.take_path_payload(id, generation, context, i);
                    self.raise_owed(payload, bag);
                }
                replay::with(|c| c.iter_mut().find(|x| x.id == context).map(|x| x.spend(i)));
            }
        }
        // A closure that catches this panic has still read a node with no
        // value, so it must not land Clean over it.
        self.dep_stack.poison_current_frame(self.runtime_id);
        if let Some(payload) = payload {
            resume_unwind(payload);
        }
        if !C::SINGLE_THREADED {
            self.note_failed_read(id);
        }
        panic!("{}", failed_read_message(id.0));
    }

    /// The payload held for the frame of `id` at `generation` at the
    /// position its compute would take, or for a frame above it whose pops
    /// reached `id`'s frame there, if a landing of `id` on another thread
    /// let that payload's read failure out. A speculative run's read takes
    /// none.
    #[cold]
    #[inline(never)]
    pub(super) fn owed_payload(&self, id: NodeId, generation: u32) -> Option<Box<dyn Any + Send>> {
        // As `take_delivery`'s scope: a closure in a speculative run can
        // catch and drop the payload the real run's read is owed.
        if replay::speculative() {
            return None;
        }
        let rt = self.runtime_id;
        let at = self.dep_stack.next_cidx();
        let candidates: Vec<(u64, usize, u64)> = replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| {
                    x.landed_payloads(at, id.0, generation)
                        .map(|(i, record)| (x.id, i, record))
                })
                .collect()
        });
        if candidates.is_empty() {
            return None;
        }
        self.park(ParkPoint::FailedReadBeforeRecheck(id));
        let found = {
            let records = self.provenance.read();
            candidates.into_iter().find(|&(_, _, record)| {
                records
                    .live
                    .iter()
                    .any(|r| r.id == record && r.propagated.contains(&(id.0, generation)))
            })
        };
        let (context, i, _) = found?;
        Some(self.take_payload(context, i, at).0)
    }

    /// The payload held for the frame of `id` at `generation` at the
    /// position its compute would take, along the read's path, for a read
    /// that found `id` Failed: a retry's rerun of an earlier read can land
    /// `id` Failed where the unbounded run leaves it Dirty for this read to
    /// compute again and raise that payload. Under `Local` and `Shared`
    /// alike, only a delivery whose read let its payload out of a Failed
    /// landing of `id` this thread stored qualifies
    /// (`Delivery::escaped`), and only when exactly one does: another
    /// thread's landing is a record's to judge. A speculative run's read
    /// takes none. Returns the delivery's context and index.
    fn path_delivery(&self, id: NodeId, generation: u32) -> Option<(u64, usize)> {
        if replay::speculative() {
            return None;
        }
        let rt = self.runtime_id;
        let at = self.dep_stack.next_cidx();
        let held: Vec<(u64, usize, u64)> = replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| {
                    x.escaped_payloads(at, id.0, generation)
                        .map(|(i, path)| (x.id, i, path))
                })
                .collect()
        });
        if held.is_empty() {
            return None;
        }
        let path = self.path_at(at);
        let mut on_path = held.into_iter().filter(|&(_, _, p)| p == path);
        // Two leave no way to tell which one the node's compute lets out.
        let (Some((context, i, _)), None) = (on_path.next(), on_path.next()) else {
            return None;
        };
        Some((context, i))
    }

    /// Takes delivery `i` of context `context`, which `path_delivery` found
    /// for a Failed read of `id`. The read stands for the node's compute,
    /// whose landing at `generation` the payload left, so a walk's entry
    /// below that contains it holds that landing too.
    fn take_path_payload(
        &self,
        id: NodeId,
        generation: u32,
        context: u64,
        i: usize,
    ) -> Box<dyn Any + Send> {
        let (payload, mut nodes) = self.take_payload(context, i, self.dep_stack.next_cidx());
        if !nodes.contains(&(id.0, generation)) {
            nodes.push((id.0, generation));
        }
        self.note_delivered(nodes, &*payload);
        payload
    }

    /// Takes the payload of delivery `i` in context `context` for a read
    /// at position `at`, with the landings its unwind made.
    fn take_payload(
        &self,
        context: u64,
        i: usize,
        at: u32,
    ) -> (Box<dyn Any + Send>, Vec<(u32, u32)>) {
        let (attributed, payload, nodes) = replay::with(|c| {
            c.iter_mut()
                .find(|x| x.id == context)
                .expect("incr-core: the delivery's context is open")
                .take_landed(i, at)
        });
        if attributed {
            cycle_state::count_failure();
        }
        (payload, nodes)
    }

    /// Raises `payload`, which [`Self::owed_payload`] took, as the read's
    /// panic, after `bag` drains in carried mode: a Stale retry's bag can
    /// still hold a displaced value, whose `Drop` must not run in the unwind.
    #[cold]
    #[inline(never)]
    pub(super) fn raise_owed(&self, payload: Box<dyn Any + Send>, bag: &mut Deferred<C>) -> ! {
        self.dep_stack.poison_current_frame(self.runtime_id);
        bag.drain_carried(payload)
    }

    /// A Failed read on this thread: if it is made for the read a record
    /// names, the read's panic is carried until a landing lets it out or
    /// another Failed read of the same node. That read's frame is the
    /// reading frame, or the virtual frame below when the read is a driver
    /// entry's own compute; the Failed node is the read node, as a walk
    /// takes a Failed dep as changed without reading it.
    fn note_failed_read(&self, id: NodeId) {
        let rt = self.runtime_id;
        if replay::carry().is_some_and(|c| c.rt == rt && c.node == id.0) {
            replay::set_carry(None);
        }
        let below = self.dep_stack.next_cidx().wrapping_sub(1);
        let reader = match self.dep_stack.active_read(rt) {
            Some((node, cidx, reading, ordinal)) if cidx == below => Some(Reader {
                node: node.0,
                generation: self.nodes.get(node.0).generation(),
                reading,
                ordinal,
            }),
            _ => replay::with(|c| {
                c.iter()
                    .filter(|x| x.rt == rt)
                    .find_map(|x| x.vframe_at(below).map(|v| ReadKey::blocked(v).into()))
            }),
        };
        let Some(reader) = reader else {
            return;
        };
        let record = self
            .provenance
            .read()
            .live
            .iter()
            .find(|r| r.reader == reader)
            .map(|r| r.id);
        if let Some(record) = record {
            replay::set_carry(Some(Carry {
                rt,
                record,
                node: id.0,
            }));
        }
    }

    /// Before `id` lands Failed, from `land_failed`, which every Failed
    /// landing goes through. The landing adds `id` to the record of a
    /// carried Failed read whose panic it lets out unchanged; and when it
    /// is the running frame's above a virtual frame blocked reading `id`,
    /// it publishes the record for the delivery that frame's unwind makes,
    /// before another thread can read `id` Failed.
    #[cold]
    #[inline(never)]
    pub(super) fn before_failed_landing(
        &self,
        id: NodeId,
        generation: u32,
        cidx: u32,
        payload: Option<&(dyn Any + Send)>,
    ) {
        let rt = self.runtime_id;
        let carried = replay::carry().filter(|c| {
            c.rt == rt
                && payload
                    .and_then(|p| p.downcast_ref::<String>())
                    .is_some_and(|m| *m == failed_read_message(c.node))
        });
        // Under the write lock before the store: a read that sees this
        // landing, or any later state of `id`, then finds the entry.
        if let Some(c) = carried {
            if !self.provenance.read().live.is_empty() {
                let mut records = self.provenance.write();
                if let Some(r) = records.live.iter_mut().find(|r| r.id == c.record) {
                    if !r.propagated.contains(&(id.0, generation)) {
                        r.propagated.push((id.0, generation));
                    }
                }
            }
        }
        if value_code::thread_word() & value_code::REPLAY_BIT == 0 || replay::speculative() {
            return;
        }
        replay::with(|c| {
            let Some((x, key)) = c.iter_mut().filter(|x| x.rt == rt).find_map(|x| {
                let key = x.landing_reader(id.0, generation, cidx)?;
                Some((x, key))
            }) else {
                return;
            };
            let record = {
                let mut records = self.provenance.write();
                let record = records.next;
                records.next += 1;
                records.live.push(Record {
                    id: record,
                    reader: key.into(),
                    propagated: Vec::new(),
                });
                record
            };
            let hold: Rc<dyn Provenance> = Rc::new(Hold::<C> {
                table: Arc::clone(&self.provenance),
                id: record,
            });
            x.provenance = Some((key, hold));
        });
    }

    /// After this thread's landing stored `id`'s frame at `cidx` Failed
    /// with `payload`, from `land_failed`: marks the deliveries for the
    /// read that let it out (`Context::mark_escaped`).
    #[cold]
    #[inline(never)]
    pub(super) fn after_failed_store(
        &self,
        id: NodeId,
        generation: u32,
        cidx: u32,
        payload: &(dyn Any + Send),
    ) {
        if value_code::thread_word() & value_code::REPLAY_BIT == 0 || replay::speculative() {
            return;
        }
        let rt = self.runtime_id;
        replay::with(|c| {
            for x in c.iter_mut().filter(|x| x.rt == rt) {
                x.mark_escaped(id.0, generation, cidx, payload);
            }
        });
    }

    /// The provenance records still held, for tests.
    #[cfg(test)]
    pub(crate) fn provenance_records(&self) -> usize {
        self.provenance.read().live.len()
    }
}
