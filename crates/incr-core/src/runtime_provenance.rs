//! Provenance of a user panic held for a virtual frame. This thread's
//! entry for Y, below which X's virtual frame reads Y, lands Y Failed and
//! holds the payload for X's retry; virtual frames hold no claim, so
//! another thread can meanwhile compute X, read Y Failed and land X
//! Failed. Only a landing that let that failed read's panic out unchanged
//! allows the serial order with this thread first, where the retry gets
//! the payload; any other allows only the other thread first, where the
//! retry reads X Failed. The record published before Y lands carries
//! which landings did.

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
    /// The nodes, as `(node, generation)`, whose latest Failed landing on
    /// another thread let the record's read failure out unchanged.
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
    /// panicked", or takes a payload this thread holds for it
    /// (`take_landed_payload`). Returns when the node has left Failed
    /// since the read saw it, to be read again.
    #[cold]
    #[inline(never)]
    pub(super) fn read_failed_node(&self, id: NodeId) {
        let mut payload = None;
        if !C::SINGLE_THREADED {
            match self.take_landed_payload(id) {
                Err(Left) => return,
                Ok(taken) => payload = taken,
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

    /// The payload held for the frame of `id` at the position its compute
    /// would take, or for a frame above it whose pops reached `id`'s frame
    /// there, if the landing of `id` this read sees let that payload's
    /// read failure out.
    fn take_landed_payload(&self, id: NodeId) -> Result<Option<Box<dyn Any + Send>>, Left> {
        let rt = self.runtime_id;
        let node = self.nodes.get(id.0);
        let generation = node.generation();
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
            return Ok(None);
        }
        let found = {
            let records = self.provenance.read();
            // Every Failed landing updates the records before its store, so
            // under the lock a node still Failed at the same generation was
            // landed by the latest update; one that left Failed, or whose
            // slot was recycled, is read again.
            if state::load::<C>(node.state_cell()) != NodeState::Failed
                || node.generation() != generation
            {
                return Err(Left);
            }
            candidates.into_iter().find(|&(_, _, record)| {
                records
                    .live
                    .iter()
                    .any(|r| r.id == record && r.propagated.contains(&(id.0, generation)))
            })
        };
        let Some((context, i, _)) = found else {
            return Ok(None);
        };
        let (attributed, payload) = replay::with(|c| {
            c.iter_mut()
                .find(|x| x.id == context)
                .expect("incr-core: the delivery's context is open")
                .take_landed(i, at)
        });
        if attributed {
            cycle_state::count_failure();
        }
        Ok(Some(payload))
    }

    /// A Failed read on this thread: if it is made for the read a record
    /// names, the read's panic is carried until a landing lets it out or
    /// another Failed read of the same node. That read's frame is the
    /// reading frame, or the virtual frame below when the read is a driver
    /// entry's own compute; the Failed node is the read node, or a dep its
    /// walk reached first.
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
    /// landing goes through. The landing replaces whatever an earlier
    /// landing of `id` recorded, adding `id` where it lets a carried
    /// Failed read's panic out unchanged; and when it is the running
    /// frame's above a virtual frame blocked reading `id`, it publishes
    /// the record for the delivery that frame's unwind makes, before
    /// another thread can read `id` Failed.
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
        // With no live record there is nothing to replace: a record
        // published after this check names no landing yet, and only this
        // landing's claim can land `id` before its store.
        if !self.provenance.read().live.is_empty() {
            let mut records = self.provenance.write();
            for r in &mut records.live {
                r.propagated.retain(|&landed| landed != (id.0, generation));
                if carried.is_some_and(|c| c.record == r.id) {
                    r.propagated.push((id.0, generation));
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

    /// The provenance records still held, for tests.
    #[cfg(test)]
    pub(crate) fn provenance_records(&self) -> usize {
        self.provenance.read().live.len()
    }
}

/// A Failed read's node left Failed, or its slot was recycled, before the
/// read could tell which landing it saw.
struct Left;
