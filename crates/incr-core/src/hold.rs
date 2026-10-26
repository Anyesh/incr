//! The landings an attempt holds under `Shared` (Variant V): a held
//! landing writes the node's own fields in place and leaves its state
//! word `Computing`, so every other thread waits on it as on a claim,
//! while this thread reads the landing through the view. The attempt's
//! end publishes every held landing, or a wound retracts them from their
//! undo records. Everything here is cold: the hot paths reach it only
//! behind [`active`] or an arm that already matched `Computing`.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::cycle::{self, ComputeCtl};
use crate::fx::FxBuildHasher;
use crate::handle::RuntimeId;
use crate::node::NodeId;
use crate::runtime::{ChildLanding, GroupInfo};
use crate::state::NodeState;
use crate::trace::TraceAction;
use crate::value_code;

/// What a node's first held landing in the attempt displaced, made once
/// and kept through the node's later re-landings. A field left `None`
/// was not written by the attempt. `changed_at` is not restored: it only
/// grows, so a retracted node's readers see a change and recompute rather
/// than verify against a value they never read.
#[derive(Default)]
pub(crate) struct Undo {
    /// The claim came from `New`, or a cut kept it after it wrote its
    /// slot, so a retraction lands it `New`; a cut's claim takes this from
    /// the cut, not from the landing.
    pub(crate) first: bool,
    /// The displaced value, boxed as the arena boxes it; `Some(None)`
    /// for a slot that was empty.
    pub(crate) value: Option<Option<Box<dyn Any>>>,
    /// `None` when the stamp was written before the entry was made, which
    /// a retraction answers with 0.
    pub(crate) verified_at: Option<u64>,
    /// With each dep's generation, so a retraction drops a dep deleted
    /// during the hold.
    pub(crate) deps: Option<Vec<(NodeId, u32)>>,
    /// The record and the `IN_GROUP` bit; for a claim a cut kept, the
    /// record from before the cut. A wound takes it back early
    /// ([`take_recs`]).
    pub(crate) rec: Option<(Option<Arc<GroupInfo>>, bool)>,
    /// The node's released-dep entry (`Inner::released_deps`).
    pub(crate) released: Option<Option<(u32, u32)>>,
}

pub(crate) struct Held {
    pub(crate) rt: RuntimeId,
    pub(crate) slot: u32,
    pub(crate) generation: u32,
    /// Landing order of the entry's last landing; publication runs in
    /// it, so a node publishes after the held nodes it read.
    seq: u64,
    pub(crate) target: NodeState,
    pub(crate) flagged: NodeState,
    pub(crate) unrun: bool,
    /// The owner claimed the node again, so the view shows its claim.
    pub(crate) reclaimed: bool,
    /// The landing's `land_children`, run at publication.
    pub(crate) children: Option<(ChildLanding, bool)>,
    /// Trace actions of the held landings, logged at publication.
    pub(crate) trace: Vec<TraceAction>,
    pub(crate) undo: Undo,
}

#[derive(Default)]
struct HoldSet {
    by_cell: HashMap<usize, usize, FxBuildHasher>,
    entries: Vec<Option<Held>>,
    /// Each runtime with held landings, so a runtime's first entry and its
    /// exit touch only its own.
    by_rt: Vec<RtHolds>,
    seq: u64,
    /// A held node was read through the view since the last claim.
    consulted: bool,
    /// What `HOLD` frames' writes displaced, by control block address
    /// (`cycle::KEPT`); `Some(None)` was an empty slot.
    kept: Vec<(usize, Option<Box<dyn Any>>)>,
    /// Every read of a held node through the view or a claim of one again,
    /// so a read can tell whether it rested on any.
    looks: u64,
    /// Episodes landed, per runtime, for `own_episode` at publication.
    episodes: Vec<(RuntimeId, u64)>,
    /// Runtimes whose runs read state this attempt has not published (a
    /// runtime whose frames are below them, or a retracted read), so their
    /// landings retract at the exit of their first entry.
    tainted: Vec<RuntimeId>,
    /// Runtimes a cycle closed in on this thread's stack: what caught it is
    /// held, so the hold stays until their first entry's exit.
    closed: Vec<RuntimeId>,
    /// Records a cut replaced on claims it keeps ([`keep_cut_rec`]).
    cut_recs: Vec<CutRec>,
    /// Runtimes with a replaced record not yet taken by [`take_recs`],
    /// so each frame of a wound's unwind finds none without a walk.
    recs: Vec<RuntimeId>,
}

/// A runtime's held landings: where its entries are, as `(entry, cell)`
/// (some taken since), and how many are still held.
struct RtHolds {
    rt: RuntimeId,
    places: Vec<(usize, usize)>,
    held: usize,
}

/// The record a cut took a claim it keeps out of, kept until the claim's
/// held landing opens its entry, which takes it as its undo, or until the
/// attempt ends or is wounded.
struct CutRec {
    rt: RuntimeId,
    slot: u32,
    rec: Option<Arc<GroupInfo>>,
    /// The claim came from `New` or wrote its slot before the cut, so a
    /// retraction lands it `New`, as a discard landing does.
    lands_new: bool,
}

impl HoldSet {
    fn note_recs(&mut self, rt: RuntimeId) {
        if !self.recs.contains(&rt) {
            self.recs.push(rt);
        }
    }

    fn get(&mut self, cell: usize) -> Option<&mut Held> {
        let at = *self.by_cell.get(&cell)?;
        self.entries[at].as_mut()
    }

    fn added(&mut self, rt: RuntimeId, at: usize, cell: usize) {
        match self.by_rt.iter_mut().find(|r| r.rt == rt) {
            Some(r) => {
                r.places.push((at, cell));
                r.held += 1;
            }
            None => self.by_rt.push(RtHolds {
                rt,
                places: vec![(at, cell)],
                held: 1,
            }),
        }
    }

    fn removed(&mut self, rt: RuntimeId) {
        let i = self
            .by_rt
            .iter()
            .position(|r| r.rt == rt)
            .expect("incr-core: a held landing's runtime is listed");
        self.by_rt[i].held -= 1;
        if self.by_rt[i].held == 0 {
            self.by_rt.swap_remove(i);
        }
    }
}

#[cfg(not(loom))]
thread_local! {
    static HOLD: RefCell<HoldSet> = RefCell::new(HoldSet::default());
}

#[cfg(loom)]
loom::thread_local! {
    static HOLD: RefCell<HoldSet> = RefCell::new(HoldSet::default());
}

fn with<R>(f: impl FnOnce(&mut HoldSet) -> R) -> R {
    HOLD.with(|h| f(&mut h.borrow_mut()))
}

/// Whether this thread holds any landing.
#[inline(always)]
pub(crate) fn active() -> bool {
    value_code::thread_word() & value_code::HELD_BIT != 0
}

/// The state the owner sees for the node whose state cell is at `cell`:
/// its held landing's target, unless the owner has claimed it again.
#[cold]
#[inline(never)]
/// [`view`] for engine bookkeeping that is no read: it marks nothing.
pub(crate) fn peek(cell: usize) -> Option<NodeState> {
    if !active() {
        return None;
    }
    with(|h| h.get(cell).filter(|e| !e.reclaimed).map(|e| e.target))
}

pub(crate) fn view(cell: usize) -> Option<NodeState> {
    if !active() {
        return None;
    }
    with(|h| {
        let t = h.get(cell).filter(|e| !e.reclaimed).map(|e| e.target)?;
        h.consulted = true;
        h.looks += 1;
        Some(t)
    })
}

/// The owner claims its own held node from `expected`, which must be the
/// target the view shows it.
#[cold]
#[inline(never)]
pub(crate) fn reclaim(cell: usize, expected: NodeState) -> bool {
    if !active() {
        return false;
    }
    with(|h| {
        let Some(e) = h.get(cell).filter(|e| !e.reclaimed && e.target == expected) else {
            return false;
        };
        e.reclaimed = true;
        h.looks += 1;
        true
    })
}

/// A count of reads that rested on a held landing, to compare across one
/// read.
pub(crate) fn looks() -> u64 {
    with(|h| h.looks)
}

/// Whether the view shows a landing for `cell`, without reading it.
pub(crate) fn shows(cell: usize) -> bool {
    active() && with(|h| h.get(cell).is_some_and(|e| !e.reclaimed))
}

/// The target of this thread's held landing for `cell`, as [`view`]
/// shows it but without counting as a look or marking the view
/// consulted.
#[cfg(all(test, not(loom)))]
pub(crate) fn target(cell: usize) -> Option<NodeState> {
    active()
        .then(|| with(|h| h.get(cell).filter(|e| !e.reclaimed).map(|e| e.target)))
        .flatten()
}

pub(crate) fn holds(cell: usize) -> bool {
    active() && with(|h| h.by_cell.contains_key(&cell))
}

/// Opens the entry for a landing about to be held, with the stamps it
/// will overwrite, unless the node already has one. `kept`, what a `HOLD`
/// frame's write displaced, goes to the undo unless an earlier write in
/// the attempt is already there; then it comes back for the caller to
/// drop.
pub(crate) fn open(
    cell: usize,
    rt: RuntimeId,
    slot: u32,
    generation: u32,
    first: bool,
    verified_at: Option<u64>,
    kept: Option<Option<Box<dyn Any>>>,
) -> Option<Box<dyn Any>> {
    value_code::set_held_bit(true);
    with(|h| {
        if let Some(e) = h.get(cell) {
            return match kept {
                Some(value) if e.undo.value.is_none() => {
                    e.undo.value = Some(value);
                    None
                }
                other => other.flatten(),
            };
        }
        let cut = h
            .cut_recs
            .iter()
            .position(|c| c.rt == rt && c.slot == slot)
            .map(|i| h.cut_recs.swap_remove(i));
        let at = h.entries.len();
        h.added(rt, at, cell);
        h.entries.push(Some(Held {
            rt,
            slot,
            generation,
            seq: 0,
            target: NodeState::Computing,
            flagged: NodeState::Computing,
            unrun: false,
            reclaimed: true,
            children: None,
            trace: Vec::new(),
            undo: Undo {
                first: cut.as_ref().map_or(first, |c| c.lands_new),
                value: kept,
                verified_at,
                deps: None,
                rec: cut.map(|c| (c.rec, true)),
                released: None,
            },
        }));
        h.by_cell.insert(cell, at);
        None
    })
}

/// The value a held node's write displaced: kept for the undo if it is
/// the first in the attempt, else handed back to drop.
pub(crate) fn keep_value(cell: usize, value: Option<Box<dyn Any>>) -> Option<Box<dyn Any>> {
    with(|h| match h.get(cell) {
        Some(e) if e.undo.value.is_none() => {
            e.undo.value = Some(value);
            None
        }
        _ => value,
    })
}

/// The dep list a held node's landing is about to replace.
pub(crate) fn keep_deps(cell: usize, old: impl FnOnce() -> Vec<(NodeId, u32)>) {
    with(|h| {
        if let Some(e) = h.get(cell).filter(|e| e.undo.deps.is_none()) {
            e.undo.deps = Some(old());
        }
    });
}

/// The released-dep entry a held node's landing is about to replace.
pub(crate) fn keep_released(cell: usize, prior: Option<(u32, u32)>) {
    with(|h| {
        if let Some(e) = h.get(cell).filter(|e| e.undo.released.is_none()) {
            e.undo.released = Some(prior);
        }
    });
}

/// The record and `IN_GROUP` bit a held node's landing is about to
/// replace.
pub(crate) fn keep_rec(cell: usize, rec: Option<Arc<GroupInfo>>, in_group: bool) {
    with(|h| {
        if let Some(e) = h.get(cell).filter(|e| e.undo.rec.is_none()) {
            e.undo.rec = Some((rec, in_group));
            let rt = e.rt;
            h.note_recs(rt);
        }
    });
}

/// Defers a held node's `land_children` to publication.
pub(crate) fn defer_children(cell: usize, landing: ChildLanding, opened: bool) {
    with(|h| {
        if let Some(e) = h.get(cell) {
            let opened = opened || e.children.is_some_and(|(_, o)| o);
            e.children = Some((landing, opened));
        }
    });
}

pub(crate) fn defer_trace(cell: usize, action: TraceAction) -> Option<TraceAction> {
    with(|h| match h.get(cell) {
        Some(e) => {
            e.trace.push(action);
            None
        }
        None => Some(action),
    })
}

/// Records a landing of a node with an entry; false if it has none.
pub(crate) fn land(cell: usize, target: NodeState, flagged: NodeState, unrun: bool) -> bool {
    with(|h| {
        h.seq += 1;
        let seq = h.seq;
        let Some(e) = h.get(cell) else {
            return false;
        };
        e.seq = seq;
        e.target = target;
        e.flagged = flagged;
        e.unrun = unrun;
        e.reclaimed = false;
        true
    })
}

/// Whether a held node was read through the view since the last claim.
pub(crate) fn consulted() -> bool {
    with(|h| h.consulted)
}

pub(crate) fn clear_consulted() {
    with(|h| h.consulted = false);
}

pub(crate) fn note_episode(rt: RuntimeId) {
    with(|h| match h.episodes.iter_mut().find(|(r, _)| *r == rt) {
        Some((_, n)) => *n += 1,
        None => h.episodes.push((rt, 1)),
    });
}

/// Every held landing, of every runtime, for the waits registry: a thread
/// waiting in one runtime may hold another's landings that its owner
/// waits on.
pub(crate) fn claims() -> Vec<(RuntimeId, u32)> {
    if !active() {
        return Vec::new();
    }
    with(|h| h.entries.iter().flatten().map(|e| (e.rt, e.slot)).collect())
}

/// The slots `rt`'s held landings occupy.
pub(crate) fn keys(rt: RuntimeId) -> Vec<u32> {
    if !active() {
        return Vec::new();
    }
    with(|h| {
        h.entries
            .iter()
            .flatten()
            .filter(|e| e.rt == rt)
            .map(|e| e.slot)
            .collect()
    })
}

/// Drops the entry of a node the owner deletes.
pub(crate) fn forget(cell: usize) -> Option<Held> {
    with(|h| {
        let at = h.by_cell.remove(&cell)?;
        let e = h.entries[at].take()?;
        h.removed(e.rt);
        Some(e)
    })
}

/// A cut takes `slot`, a claim it keeps, out of `rec`, its group: under
/// `Shared` the claim's landing is held ([`cut_kept`]), so that a
/// retraction puts the whole group back, and a wound puts the record back
/// before its unwind lands anything ([`take_recs`]).
pub(crate) fn keep_cut_rec(rt: RuntimeId, slot: u32, rec: Option<Arc<GroupInfo>>, lands_new: bool) {
    value_code::set_held_bit(true);
    with(|h| {
        if !h.cut_recs.iter().any(|c| c.rt == rt && c.slot == slot) {
            h.cut_recs.push(CutRec {
                rt,
                slot,
                rec,
                lands_new,
            });
            h.note_recs(rt);
        }
    });
}

/// Whether a cut took `slot`, a claim of `rt` that has not landed held
/// since, out of its group.
pub(crate) fn cut_kept(rt: RuntimeId, slot: u32) -> bool {
    active() && with(|h| h.cut_recs.iter().any(|c| c.rt == rt && c.slot == slot))
}

/// Takes the records and `IN_GROUP` bits `rt`'s attempt replaced, held
/// landings' and cuts' alike, for a wound to put back before its unwind
/// lands anything: a landing published first could name a member whose
/// record is still the attempt's. The rest of each entry retracts at the
/// entry's exit.
pub(crate) fn take_recs(rt: RuntimeId) -> Vec<(u32, Option<Arc<GroupInfo>>, bool)> {
    with(|h| {
        let Some(at) = h.recs.iter().position(|&r| r == rt) else {
            return Vec::new();
        };
        h.recs.swap_remove(at);
        let mut recs: Vec<_> = h
            .entries
            .iter_mut()
            .flatten()
            .filter(|e| e.rt == rt)
            .filter_map(|e| {
                e.undo
                    .rec
                    .take()
                    .map(|(rec, in_group)| (e.slot, rec, in_group))
            })
            .collect();
        recs.extend(
            take_cut_recs(h, rt)
                .into_iter()
                .map(|(slot, rec)| (slot, rec, true)),
        );
        recs
    })
}

fn take_cut_recs(h: &mut HoldSet, rt: RuntimeId) -> Vec<(u32, Option<Arc<GroupInfo>>)> {
    let mut taken = Vec::new();
    h.cut_recs.retain_mut(|c| {
        if c.rt == rt {
            taken.push((c.slot, c.rec.take()));
        }
        c.rt != rt
    });
    taken
}

/// What an attempt's end takes for one runtime.
pub(crate) struct Taken {
    /// In landing order.
    pub(crate) entries: Vec<Held>,
    pub(crate) episodes: u64,
    /// The records of cut claims whose landing opened no entry, which a
    /// retraction puts back and a publication leaves as the cut left them.
    pub(crate) cut_recs: Vec<(u32, Option<Arc<GroupInfo>>)>,
}

/// Takes `rt`'s entries, with the episodes it landed, for publication or
/// retraction, and clears its taint.
pub(crate) fn take(rt: RuntimeId) -> Taken {
    with(|h| {
        let mut taken: Vec<Held> = Vec::new();
        visited(h.by_rt.len());
        if let Some(i) = h.by_rt.iter().position(|r| r.rt == rt) {
            let places = h.by_rt.swap_remove(i).places;
            visited(places.len());
            for (at, cell) in places {
                if let Some(e) = h.entries[at].take() {
                    h.by_cell.remove(&cell);
                    taken.push(e);
                }
            }
        }
        h.tainted.retain(|r| *r != rt);
        h.closed.retain(|r| *r != rt);
        h.recs.retain(|r| *r != rt);
        let cut_recs = take_cut_recs(h, rt);
        // A taint keeps the bit, so that the `HOLD` it raised passes up.
        if h.by_cell.is_empty()
            && h.tainted.is_empty()
            && h.closed.is_empty()
            && h.cut_recs.is_empty()
        {
            h.entries.clear();
            debug_assert!(h.by_rt.is_empty());
            h.consulted = false;
            value_code::set_held_bit(false);
        }
        taken.sort_by_key(|e| e.seq);
        let episodes = h
            .episodes
            .iter()
            .position(|(r, _)| *r == rt)
            .map_or(0, |i| h.episodes.swap_remove(i).1);
        Taken {
            entries: taken,
            episodes,
            cut_recs,
        }
    })
}

/// Whether this thread holds landings of `rt`, or has tainted it.
pub(crate) fn holds_runtime(rt: RuntimeId) -> bool {
    active()
        && with(|h| {
            visited(h.tainted.len() + h.closed.len() + h.by_rt.len() + h.cut_recs.len());
            h.tainted.contains(&rt)
                || h.closed.contains(&rt)
                || h.by_rt.iter().any(|r| r.rt == rt)
                || h.cut_recs.iter().any(|c| c.rt == rt)
        })
}

/// `rt`'s runs read state this attempt has not published.
pub(crate) fn taint(rt: RuntimeId) {
    value_code::set_held_bit(true);
    with(|h| {
        if !h.tainted.contains(&rt) {
            h.tainted.push(rt);
        }
    });
}

/// A cycle closed in `rt` on this thread's stack.
pub(crate) fn note_closed(rt: RuntimeId) {
    value_code::set_held_bit(true);
    with(|h| {
        if !h.closed.contains(&rt) {
            h.closed.push(rt);
        }
    });
}

pub(crate) fn closed(rt: RuntimeId) -> bool {
    active() && with(|h| h.closed.contains(&rt))
}

pub(crate) fn tainted(rt: RuntimeId) -> bool {
    active() && with(|h| h.tainted.contains(&rt))
}

#[cfg(all(test, not(loom)))]
thread_local! {
    static VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Counts places a runtime's check or take went over, for tests.
fn visited(n: usize) {
    #[cfg(all(test, not(loom)))]
    VISITS.with(|v| v.set(v.get() + n));
    #[cfg(not(all(test, not(loom))))]
    let _ = n;
}

/// The places [`visited`] counted on this thread since the last call.
#[cfg(all(test, not(loom)))]
pub(crate) fn take_visits() -> usize {
    VISITS.with(|v| v.replace(0))
}

/// How many nodes this thread holds now.
#[cfg(all(test, not(loom)))]
pub(crate) fn held_count() -> usize {
    with(|h| h.by_cell.len())
}

/// A `HOLD` frame's write keeps what it displaced until the frame's landing
/// opens its entry; a second write of the frame drops the first one's.
pub(crate) fn keep_for(ctl: &ComputeCtl, displaced: Option<Box<dyn Any>>) {
    let key = std::ptr::from_ref(ctl) as usize;
    ctl.raise(cycle::KEPT);
    let old = with(|h| match h.kept.iter_mut().find(|(k, _)| *k == key) {
        Some((_, v)) => Some(std::mem::replace(v, displaced)),
        None => {
            h.kept.push((key, displaced));
            None
        }
    });
    drop(old);
}

/// What `ctl`'s frame kept, for its landing's entry.
pub(crate) fn take_kept(ctl: &ComputeCtl) -> Option<Option<Box<dyn Any>>> {
    if !ctl.has(cycle::KEPT) {
        return None;
    }
    ctl.flags.set(ctl.flags.get() & !cycle::KEPT);
    let key = std::ptr::from_ref(ctl) as usize;
    with(|h| {
        let at = h.kept.iter().position(|(k, _)| *k == key)?;
        Some(h.kept.swap_remove(at).1)
    })
}

/// Drops what a compute frame kept and no entry took, when the frame
/// ends, as an ordinary write drops what it displaced; the key is unique
/// while the control block lives.
pub(crate) struct KeptGuard<'a>(pub(crate) &'a ComputeCtl);

impl Drop for KeptGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.0.has(cycle::KEPT) {
            drop(take_kept(self.0));
        }
    }
}
