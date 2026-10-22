//! Provenance of a user panic held for a virtual frame. This thread's
//! entry for Y, below which X's virtual frame reads Y, lands Y Failed and
//! holds the payload for X's retry; virtual frames hold no claim, so
//! another thread can meanwhile compute X, read Y Failed and land X
//! Failed. Only a landing that let that failed read's panic out unchanged
//! allows the serial order with this thread first, where the retry gets
//! the payload; any other allows only the other thread first, where the
//! retry reads X Failed. The record published before Y lands carries
//! which landings did.

use std::panic::resume_unwind;
use std::rc::Rc;
use std::sync::Arc;

use super::replay_impl::copy_payload;
use super::*;
use crate::locks::Lock;
use crate::replay::{self, Carry, Delivered, Provenance};

/// The read a record is for: the virtual frame's node and generation,
/// the node it was blocked reading and which of its reads of it that was.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Reader {
    node: u32,
    generation: u32,
    reading: u64,
    ordinal: u32,
}

struct Record {
    id: u64,
    reader: Reader,
    /// The landings, as `(node, generation)`, of runs on other threads
    /// that let the record's read failure out unchanged.
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
    fn propagated(&self, node: u32, generation: u32) -> bool {
        self.table
            .read()
            .live
            .iter()
            .find(|r| r.id == self.id)
            .is_some_and(|r| r.propagated.contains(&(node, generation)))
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
    /// `compute_one`'s Failed arm: the read panics "last compute
    /// panicked", unless this thread holds a payload for the frame the
    /// node's compute would run in and the node's landing let that
    /// payload's read failure out, in which case the read takes the payload.
    #[cold]
    #[inline(never)]
    pub(super) fn read_failed_node(&self, id: NodeId) -> ! {
        // A closure that catches this panic has still read a node with no
        // value, so it must not land Clean over it.
        self.dep_stack.poison_current_frame(self.runtime_id);
        if !C::SINGLE_THREADED {
            if let Some(payload) = self.take_landed_payload(id) {
                resume_unwind(payload);
            }
            self.note_failed_read(id);
        }
        panic!("{}", failed_read_message(id.0));
    }

    /// The payload held for the frame of `id` at the position its compute
    /// would take, or for a frame above it that popped with `id`'s frame
    /// there, if another thread's landing of `id` let that payload's read
    /// failure out. Taken as `take_delivery` takes one: a copy, and the
    /// delivery kept for a retry doomed again, if the payload can be
    /// copied. When the read is the running entry's own compute, the
    /// entry's unwind carries the record on to the frame below.
    fn take_landed_payload(&self, id: NodeId) -> Option<Box<dyn std::any::Any + Send>> {
        let rt = self.runtime_id;
        let generation = self.nodes.get(id.0).generation();
        let at = self.dep_stack.next_cidx();
        let (attributed, payload) = replay::with(|c| {
            c.iter_mut().filter(|x| x.rt == rt).find_map(|x| {
                let popped = |cidx, node, generation| x.popped.contains(&(cidx, node, generation));
                let i = x.pending.iter().position(|d| {
                    let k = d.key;
                    !d.taken
                        && matches!(d.what, Delivered::Unwind { cycle: false, .. })
                        && ((k.cidx == at && k.node == id.0 && k.generation == generation)
                            || (k.cidx > at
                                && popped(at, id.0, generation)
                                && popped(k.cidx, k.node, k.generation)))
                        && d.provenance
                            .as_ref()
                            .is_some_and(|p| p.propagated(id.0, generation))
                })?;
                let key = x.pending[i].key;
                for d in x.pending.iter_mut().filter(|d| d.key == key) {
                    d.taken = true;
                }
                if x.running.is_some_and(|level| x.boundary + level == at) {
                    x.provenance = x.pending[i].provenance.clone();
                }
                let Delivered::Unwind {
                    attributed,
                    payload,
                    ..
                } = &x.pending[i].what
                else {
                    unreachable!("incr-core: matched as an unwind");
                };
                let attributed = *attributed;
                let payload = match copy_payload(&**payload) {
                    Some(copy) => copy,
                    None => match x.pending.remove(i).what {
                        Delivered::Unwind { payload, .. } => payload,
                        _ => unreachable!("incr-core: matched as an unwind"),
                    },
                };
                Some((attributed, payload))
            })
        })?;
        if attributed {
            cycle_state::count_failure();
        }
        Some(payload)
    }

    /// A Failed read on this thread: if it is made for the read a record
    /// names, the read's panic is carried until a landing lets it out or
    /// another Failed read of the same node. That read's frame is the
    /// reading frame, or the virtual frame below when the read is a driver
    /// entry's own compute, whose retry the panic is delivered to; the
    /// Failed node is the read node, or a dep its walk reached first.
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
                c.iter().filter(|x| x.rt == rt).find_map(|x| {
                    let v = x.vstack.get(below.checked_sub(x.boundary)? as usize)?;
                    (v.cidx == below).then_some(Reader {
                        node: v.node,
                        generation: v.generation,
                        reading: v.reading,
                        ordinal: v.ordinal,
                    })
                })
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

    /// Before `id` lands Failed from a user panic: if `id` is the running
    /// entry's node above a virtual frame, its record is published for
    /// the delivery the entry's unwind makes, before the landing another
    /// thread's read can see; and if the panic is a carried Failed read's,
    /// unchanged, this landing is recorded as letting it out.
    #[cold]
    #[inline(never)]
    pub(super) fn before_failed_landing(
        &self,
        id: NodeId,
        generation: u32,
        cidx: u32,
        payload: &(dyn std::any::Any + Send),
    ) {
        if C::SINGLE_THREADED {
            return;
        }
        let rt = self.runtime_id;
        if let Some(carry) = replay::carry().filter(|c| c.rt == rt) {
            if payload
                .downcast_ref::<String>()
                .is_some_and(|m| *m == failed_read_message(carry.node))
            {
                if let Some(r) = self
                    .provenance
                    .write()
                    .live
                    .iter_mut()
                    .find(|r| r.id == carry.record)
                {
                    r.propagated.push((id.0, generation));
                }
            }
        }
        if value_code::thread_word() & value_code::REPLAY_BIT == 0 {
            return;
        }
        let reader = replay::with(|c| {
            c.iter().filter(|x| x.rt == rt).find_map(|x| {
                let level = x.running?;
                let entry = x.running_entry?;
                let below = (level > 0
                    && !entry.spec
                    && entry.node == id.0
                    && entry.generation == generation
                    && x.boundary + level == cidx)
                    .then(|| x.vstack.get(level as usize - 1))
                    .flatten()?;
                Some(Reader {
                    node: below.node,
                    generation: below.generation,
                    reading: below.reading,
                    ordinal: below.ordinal,
                })
            })
        });
        let Some(reader) = reader else {
            return;
        };
        let id_of = {
            let mut records = self.provenance.write();
            let record = records.next;
            records.next += 1;
            records.live.push(Record {
                id: record,
                reader,
                propagated: Vec::new(),
            });
            record
        };
        let hold: Rc<dyn Provenance> = Rc::new(Hold::<C> {
            table: Arc::clone(&self.provenance),
            id: id_of,
        });
        replay::with(|c| {
            if let Some(x) = c
                .iter_mut()
                .find(|x| x.rt == rt && x.running.is_some_and(|level| x.boundary + level == cidx))
            {
                x.provenance = Some(hold);
            }
        });
    }

    /// The provenance records still held, for tests.
    #[cfg(test)]
    pub(crate) fn provenance_records(&self) -> usize {
        self.provenance.read().live.len()
    }
}
