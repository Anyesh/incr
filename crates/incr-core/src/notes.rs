//! A thread's notes: the Failed landings its panics in flight made, and
//! the landings its reads raised again, each at its frame position and
//! by the payload it was noted for (`runtime_contain`).
//!
//! A frame that catches many panics keeps a note per panic, and a read
//! that raises one keeps a copy of every note of its payload. Each
//! payload's notes are a log per runtime that the copies share, so a copy
//! costs nothing per note, and the positions each payload has notes at are
//! indexed, so retiring notes above a frame visits only the payloads it
//! retires. A read that takes a ticket notes it as one note, which a
//! walk's entry that contains the payload again holds as the same ticket.

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::{btree_map, BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::fx::FxBuildHasher;
use crate::handle::RuntimeId;
use crate::replay::{note_work, Landings, PayloadKey, SharedCopy};

/// Removals of notes and of a ticket's landings, on every thread: a copy
/// taken at `EPOCH` sees what a later removal takes.
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// The number a removal marks what it takes with.
pub(crate) fn removal() -> u64 {
    EPOCH.fetch_add(1, SeqCst)
}

/// The number a copy taken now sees removals from.
pub(crate) fn epoch() -> u64 {
    EPOCH.load(SeqCst)
}

/// A ticket's landings in the order they joined it, shared by the note a
/// read that takes the ticket leaves and by the copies taken of the note.
pub(crate) struct Members {
    /// The ticket's id.
    pub(crate) id: u64,
    joined: Mutex<Joined>,
    /// The tickets whose read's note went without holding them again, for
    /// their runtime to let go.
    dropped: Arc<Mutex<Vec<u64>>>,
}

#[derive(Default)]
pub(crate) struct Joined {
    pub(crate) members: Vec<Member>,
    /// As `Log::marked`.
    pub(crate) marked: Marked,
}

/// How far a set's generation has marked a list of notes: every one
/// before `upto` still noted at removal `epoch` is in the set.
#[derive(Clone, Copy, Default)]
pub(crate) struct Marked {
    generation: u64,
    upto: usize,
    epoch: u64,
}

impl Marked {
    /// Where a mark of the first `len` notes still noted at removal `epoch`
    /// for `generation` starts, moving the watermark past them. A copy
    /// taken before a later one holds notes removed in between, which the
    /// later one's mark left out, so it marks from the start.
    pub(crate) fn advance(&mut self, generation: u64, len: usize, epoch: u64) -> usize {
        if self.generation == generation && epoch >= self.epoch {
            let from = self.upto.min(len);
            if len > self.upto {
                self.upto = len;
                self.epoch = epoch;
            }
            from
        } else {
            *self = Marked {
                generation,
                upto: len,
                epoch,
            };
            0
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Member {
    pub(crate) node: (u32, u32),
    /// As `Noted::at`.
    pub(crate) at: u64,
    /// As `Noted::gone`.
    pub(crate) gone: u64,
}

impl Members {
    pub(crate) fn new(id: u64, dropped: Arc<Mutex<Vec<u64>>>) -> Self {
        Self {
            id,
            joined: Mutex::default(),
            dropped,
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, Joined> {
        self.joined.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn drop_note(&self) {
        self.dropped
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(self.id);
    }
}

/// One runtime's notes of one payload, in the order noted. A note leaves
/// by the number of the removal that took it, not from the log, so that a
/// copy taken before still finds it.
#[derive(Default)]
pub(crate) struct Log {
    pub(crate) entries: Vec<Noted>,
    /// How far the copied set has marked the notes copies took
    /// (`replay::mark_copied`).
    pub(crate) marked: Marked,
    /// As `marked`, for the kept reads' copies (`replay::keep`).
    pub(crate) kept_marked: Marked,
    /// The copy of this log [`Notes::note_copy`] noted last.
    renoted: Option<Renoted>,
}

#[derive(Clone)]
pub(crate) struct Noted {
    pub(crate) node: (u32, u32),
    /// The stamp of the Failed landing of `node` the note names: a later
    /// landing of the node, on any thread, has another.
    pub(crate) at: u64,
    /// A ticket a read took, whose first members the note stands for in
    /// place of `node`.
    pub(crate) lent: Option<(Arc<Members>, usize)>,
    pub(crate) cidx: u32,
    /// Taken from a ticket by a read: a discarded run's read was no
    /// reader's observation, so its drop holds the ticket again.
    restorable: bool,
    /// The number of the removal that took the note, `u64::MAX` while it
    /// is noted.
    pub(crate) gone: u64,
}

/// What a catch-up took from a shared copy without taking it apart: the
/// landings deeper than `depth`, and those of the nodes `landed` has a
/// landing of from event `seen` on.
#[derive(Clone)]
pub(crate) struct Cut {
    pub(crate) depth: u32,
    pub(crate) seen: usize,
    pub(crate) landed: Rc<CutLandings>,
}

/// The landings logged while a frame's reads were kept: each node's
/// latest, and all of them in log order.
#[derive(Default)]
pub(crate) struct CutLandings {
    pub(crate) latest: HashMap<(u32, u32), usize, FxBuildHasher>,
    pub(crate) order: Vec<usize>,
}

impl Cut {
    fn keeps(&self, node: (u32, u32), depth: u32) -> bool {
        depth <= self.depth && self.landed.latest.get(&node).is_none_or(|&i| i < self.seen)
    }

    /// Whether `self` keeps exactly the entries `other` keeps: the same
    /// depth, and no landing logged between the two `seen`.
    fn same_as(&self, other: &Cut) -> bool {
        let (lo, hi) = (self.seen.min(other.seen), self.seen.max(other.seen));
        let order = &self.landed.order;
        Rc::ptr_eq(&self.landed, &other.landed)
            && self.depth == other.depth
            && order.partition_point(|&i| i < hi) == order.partition_point(|&i| i < lo)
    }
}

impl Log {
    /// The landings of the first `len` notes still noted at removal
    /// `epoch`, one per node and depth above `base`, with the newest stamp
    /// noted there, but those `cut` took. A note below `base` counts as
    /// just above it. A node keeps a landing at each depth it was noted at,
    /// since each depth's stamp goes when that frame's notes do.
    pub(crate) fn landings(
        &self,
        len: usize,
        epoch: u64,
        base: u32,
        cut: Option<&Cut>,
    ) -> Vec<Landed> {
        self.landings_in(0, len, epoch, base, cut)
    }

    /// As [`Self::landings`], of the notes from entry `from` on only.
    fn landings_in(
        &self,
        from: usize,
        len: usize,
        epoch: u64,
        base: u32,
        cut: Option<&Cut>,
    ) -> Vec<Landed> {
        note_work(len - from);
        let mut nodes: Vec<Landed> = Vec::new();
        let mut at: HashMap<((u32, u32), u32), usize, FxBuildHasher> = HashMap::default();
        let mut add = |node: (u32, u32), depth: u32, stamp: u64| {
            if cut.is_some_and(|c| !c.keeps(node, depth)) {
                return;
            }
            match at.entry((node, depth)) {
                Entry::Occupied(i) => {
                    let s = &mut nodes[*i.get()].2;
                    *s = (*s).max(stamp);
                }
                Entry::Vacant(v) => {
                    v.insert(nodes.len());
                    nodes.push((node, depth, stamp));
                }
            }
        };
        for e in self.entries[from..len].iter().filter(|e| e.gone >= epoch) {
            let depth = e.cidx.saturating_sub(base);
            match &e.lent {
                None => add(e.node, depth, e.at),
                Some((members, len)) => {
                    note_work(*len);
                    for m in members.lock().members[..*len]
                        .iter()
                        .filter(|m| m.gone >= epoch)
                    {
                        add(m.node, depth, m.at);
                    }
                }
            }
        }
        nodes
    }
}

/// A noted landing, its depth above the reading frame and its stamp
/// (`Noted::at`).
pub(crate) type Landed = ((u32, u32), u32, u64);

/// A noted landing and its stamp (`Noted::at`).
pub(crate) type Stamped = ((u32, u32), u64);

/// A note's place: entry `idx` of log `log` of group `group`.
#[derive(Clone, Copy)]
struct At {
    group: u64,
    log: u32,
    idx: u32,
}

/// The notes of one payload.
struct Group {
    /// Its id, for the places of its notes, which outlast its payload's
    /// key when it is parked and goes back under another copy's key.
    id: u64,
    logs: Vec<RuntimeLog>,
    /// The positions with a note, with the notes there, some of which may
    /// have gone since.
    levels: BTreeMap<u32, Level>,
    /// The notes still noted, the sum of every level's `live`.
    live: usize,
    /// A read's ticket was noted in it.
    lent: bool,
}

struct RuntimeLog {
    rt: RuntimeId,
    log: Rc<RefCell<Log>>,
    /// The notes before this one were copied, so none is restorable.
    copied: usize,
}

#[derive(Default)]
struct Level {
    live: u32,
    at: Vec<(u32, u32)>,
}

impl Group {
    fn top(&self) -> Option<u32> {
        self.levels.last_key_value().map(|(&cidx, _)| cidx)
    }

    fn log(&self, rt: RuntimeId) -> Option<&RuntimeLog> {
        self.logs.iter().find(|l| l.rt == rt)
    }

    /// Takes note `idx` of log `log` out at removal `at`, returning it if
    /// it was still noted.
    fn take(&self, log: u32, idx: u32, at: u64) -> Option<Noted> {
        let mut l = self.logs[log as usize].log.borrow_mut();
        let e = &mut l.entries[idx as usize];
        (e.gone == u64::MAX).then(|| {
            e.gone = at;
            e.clone()
        })
    }
}

pub(crate) struct Notes {
    groups: BTreeMap<PayloadKey, Group>,
    /// Each payload's highest position with a note.
    tops: BTreeSet<(u32, PayloadKey)>,
    /// Where each landing was noted. A place may have been taken since, or
    /// hold another note after its payload's notes all went.
    by_node: BTreeMap<(RuntimeId, (u32, u32)), Vec<At>>,
    live: usize,
    /// Bumped when a note goes, which voids every log's `renoted`.
    renote_gen: u64,
    /// The groups [`Self::take`] took whole, each kept to go back as it was
    /// when the copy it handed out is delivered at the place it was taken
    /// from. `by_node` keeps their places meanwhile.
    parked: Vec<Parked>,
    /// Each group's key by its id.
    keys: BTreeMap<u64, PayloadKey>,
    next_group: u64,
}

/// A group [`Notes::take`] parked, with the removal `epoch` and `base` of
/// the copy it handed out.
struct Parked {
    group: Group,
    epoch: u64,
    base: u32,
}

/// The copy a log's last [`Notes::note_copy`] noted, whose first `len`
/// entries a later copy to the same place need not note again.
struct Renoted {
    renote_gen: u64,
    len: usize,
    epoch: u64,
    from_base: u32,
    rt: RuntimeId,
    key: PayloadKey,
    base: u32,
    cut: Option<Cut>,
}

impl Notes {
    pub(crate) const fn new() -> Self {
        Self {
            groups: BTreeMap::new(),
            tops: BTreeSet::new(),
            by_node: BTreeMap::new(),
            live: 0,
            renote_gen: 0,
            parked: Vec::new(),
            keys: BTreeMap::new(),
            next_group: 0,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.live == 0
    }

    pub(crate) fn push(
        &mut self,
        rt: RuntimeId,
        key: PayloadKey,
        cidx: u32,
        (node, at): Stamped,
        restorable: bool,
    ) {
        self.push_noted(rt, key, cidx, (node, at), None, restorable);
    }

    /// Notes, under `key` and above `base`, the landings `copy` holds: the
    /// first `len` entries of its log still noted at removal `epoch`, at
    /// their depths above the copy's base, but those its cut took. When
    /// the copy of the log noted last went to the same place, at an epoch
    /// no later, under the same cut, and no note has gone since, the
    /// entries it held are noted already, so only later ones are: a second
    /// note of a node at the same place, or deeper than another, retires
    /// with that one and adds no landing.
    pub(crate) fn note_copy(
        &mut self,
        rt: RuntimeId,
        key: PayloadKey,
        base: u32,
        copy: SharedCopy<'_>,
    ) {
        if self.unpark(rt, key, base, &copy) {
            return;
        }
        let SharedCopy {
            log,
            len,
            epoch,
            base: from_base,
            cut,
        } = copy;
        let from = match &log.borrow().renoted {
            Some(r)
                if (r.renote_gen, r.rt, r.key, r.base, r.from_base)
                    == (self.renote_gen, rt, key, base, from_base)
                    && r.len <= len
                    && r.epoch <= epoch
                    && match (&r.cut, cut) {
                        (None, None) => true,
                        (Some(a), Some(b)) => a.same_as(b),
                        _ => false,
                    } =>
            {
                r.len
            }
            _ => 0,
        };
        let nodes = log.borrow().landings_in(from, len, epoch, from_base, cut);
        for (node, depth, at) in nodes {
            self.push(rt, key, base + depth, (node, at), false);
        }
        log.borrow_mut().renoted = Some(Renoted {
            renote_gen: self.renote_gen,
            len,
            epoch,
            from_base,
            rt,
            key,
            base,
            cut: cut.cloned(),
        });
    }

    /// Puts the group parked with `copy` back under `key`, as noting
    /// `copy` again above `base` would, when `copy` is the one its take
    /// handed out, delivered where it was taken, and no note of its log
    /// came or went since. Any other copy of the log drops it.
    fn unpark(&mut self, rt: RuntimeId, key: PayloadKey, base: u32, copy: &SharedCopy<'_>) -> bool {
        if self.parked.is_empty() {
            return false;
        }
        note_work(self.parked.len());
        let Some(i) = self
            .parked
            .iter()
            .position(|p| Rc::ptr_eq(&p.group.logs[0].log, copy.log))
        else {
            return false;
        };
        let Parked {
            group: g,
            epoch,
            base: from,
        } = self.parked.swap_remove(i);
        let len = copy.log.borrow().entries.len();
        let as_taken = g.logs[0].rt == rt
            && (from, copy.base) == (base, base)
            && (copy.epoch, copy.len) == (epoch, len)
            && copy.cut.is_none();
        if !as_taken || self.groups.contains_key(&key) {
            return false;
        }
        let top = g.top();
        self.live += g.live;
        self.keys.insert(g.id, key);
        g.logs[0].log.borrow_mut().renoted = Some(Renoted {
            renote_gen: self.renote_gen,
            len,
            epoch,
            from_base: base,
            rt,
            key,
            base,
            cut: None,
        });
        self.groups.insert(key, g);
        self.retop(key, None, top);
        true
    }

    /// Notes the ticket `members` stands for, which a read took with the
    /// payload of `key`, as its members so far.
    pub(crate) fn push_lent(
        &mut self,
        rt: RuntimeId,
        key: PayloadKey,
        cidx: u32,
        members: Arc<Members>,
    ) {
        let len = members.lock().members.len();
        self.push_noted(
            rt,
            key,
            cidx,
            ((u32::MAX, 0), 0),
            Some((members, len)),
            true,
        );
    }

    fn push_noted(
        &mut self,
        rt: RuntimeId,
        key: PayloadKey,
        cidx: u32,
        (node, at): Stamped,
        lent: Option<(Arc<Members>, usize)>,
        restorable: bool,
    ) {
        note_work(1);
        let indexed = lent.is_none();
        let g = match self.groups.entry(key) {
            btree_map::Entry::Occupied(g) => g.into_mut(),
            btree_map::Entry::Vacant(v) => {
                let id = self.next_group;
                self.next_group += 1;
                self.keys.insert(id, key);
                v.insert(Group {
                    id,
                    logs: Vec::new(),
                    levels: BTreeMap::new(),
                    live: 0,
                    lent: false,
                })
            }
        };
        g.live += 1;
        g.lent |= !indexed;
        let old = g.top();
        let log = match g.logs.iter().position(|l| l.rt == rt) {
            Some(i) => i,
            None => {
                g.logs.push(RuntimeLog {
                    rt,
                    log: Rc::default(),
                    copied: 0,
                });
                g.logs.len() - 1
            }
        };
        let idx = {
            let mut l = g.logs[log].log.borrow_mut();
            l.entries.push(Noted {
                node,
                at,
                lent,
                cidx,
                restorable,
                gone: u64::MAX,
            });
            l.entries.len() - 1
        };
        let (log, idx) = (log as u32, idx as u32);
        let level = g.levels.entry(cidx).or_default();
        level.live += 1;
        level.at.push((log, idx));
        let (new, group) = (g.top(), g.id);
        self.retop(key, old, new);
        if indexed {
            self.by_node
                .entry((rt, node))
                .or_default()
                .push(At { group, log, idx });
        }
        self.live += 1;
    }

    fn retop(&mut self, key: PayloadKey, old: Option<u32>, new: Option<u32>) {
        if old != new {
            if let Some(top) = old {
                self.tops.remove(&(top, key));
            }
            if let Some(top) = new {
                self.tops.insert((top, key));
            }
        }
    }

    fn removed(&mut self) {
        // A parked group whose copy went can never go back.
        self.parked
            .retain(|p| Rc::strong_count(&p.group.logs[0].log) > 1);
        if self.live == 0 && self.parked.is_empty() {
            self.by_node.clear();
        }
    }

    /// Removes `rt`'s notes of `node`'s landing.
    pub(crate) fn forget_node(&mut self, rt: RuntimeId, node: (u32, u32)) {
        let Some(places) = self.by_node.remove(&(rt, node)) else {
            return;
        };
        note_work(places.len());
        let removal = removal();
        for at in places {
            // A parked group's places are no longer all indexed, so it goes
            // back only as its copy's notes.
            if !self.parked.is_empty() {
                note_work(self.parked.len());
                self.parked.retain(|p| p.group.id != at.group);
            }
            let Some(&key) = self.keys.get(&at.group) else {
                continue;
            };
            let g = self
                .groups
                .get_mut(&key)
                .expect("incr-core: a group's key is indexed");
            let Some(l) = g.logs.get(at.log as usize).filter(|l| l.rt == rt) else {
                continue;
            };
            let cidx = match l.log.borrow().entries.get(at.idx as usize) {
                Some(e) if e.node == node && e.gone == u64::MAX => e.cidx,
                _ => continue,
            };
            g.take(at.log, at.idx, removal);
            let old = g.top();
            let level = g
                .levels
                .get_mut(&cidx)
                .expect("incr-core: a note's level is indexed");
            level.live -= 1;
            g.live -= 1;
            if level.live == 0 {
                g.levels.remove(&cidx);
            }
            let new = g.top();
            if new.is_none() {
                self.groups.remove(&key);
                self.keys.remove(&at.group);
            }
            self.live -= 1;
            self.renote_gen += 1;
            self.retop(key, old, new);
        }
        self.removed();
    }

    /// Removes the notes at `from` and above, but those of `keep`.
    pub(crate) fn forget_from(&mut self, from: u32, keep: Option<PayloadKey>) {
        let keys: Vec<PayloadKey> = self
            .tops
            .iter()
            .rev()
            .take_while(|&&(top, _)| top >= from)
            .filter(|&&(_, key)| Some(key) != keep)
            .map(|&(_, key)| key)
            .collect();
        note_work(keys.len() + 1);
        for key in keys {
            self.forget_levels(key, from, None);
        }
        self.removed();
    }

    /// Removes `key`'s notes at `from` and above, returning those of `rt`
    /// as (entry, landing), and the tickets its reads took.
    fn forget_levels(&mut self, key: PayloadKey, from: u32, rt: Option<RuntimeId>) -> Taken {
        let mut out = Taken::default();
        let Some(g) = self.groups.get_mut(&key) else {
            return out;
        };
        let old = g.top();
        let gone = g.levels.split_off(&from);
        let mine = rt.and_then(|rt| g.logs.iter().position(|l| l.rt == rt));
        let removal = removal();
        for (_, level) in gone {
            note_work(level.at.len());
            self.live -= level.live as usize;
            g.live -= level.live as usize;
            self.renote_gen += 1;
            for (log, idx) in level.at {
                let Some(e) = g.take(log, idx, removal) else {
                    continue;
                };
                match (e.lent, mine == Some(log as usize)) {
                    (None, true) => out.nodes.push((idx, (e.node, e.at))),
                    (None, false) => {}
                    (Some((members, _)), true) => out.lent.push(members),
                    (Some((members, _)), false) => members.drop_note(),
                }
            }
        }
        let (new, id) = (g.top(), g.id);
        if new.is_none() {
            self.groups.remove(&key);
            self.keys.remove(&id);
        }
        self.retop(key, old, new);
        out
    }

    /// A walk's entry at `cidx` contains `key`'s payload: the notes above
    /// `cidx` go, and `key`'s at it. Returns `rt`'s landings of `key` that
    /// went, in the order noted, and the tickets its reads took.
    pub(crate) fn contain(
        &mut self,
        rt: RuntimeId,
        cidx: u32,
        key: PayloadKey,
    ) -> (Vec<Stamped>, Vec<Arc<Members>>) {
        let mut ours = self.forget_levels(key, cidx, Some(rt));
        if let Some(above) = cidx.checked_add(1) {
            self.forget_from(above, None);
        } else {
            self.removed();
        }
        ours.nodes.sort_unstable_by_key(|&(idx, _)| idx);
        (newest(ours.nodes.into_iter().map(|(_, n)| n)), ours.lent)
    }

    /// A copy of `rt`'s notes of `key`, at their depths above `base`. The
    /// notes of `key` it copies are no longer restorable.
    pub(crate) fn copy(&mut self, rt: RuntimeId, key: PayloadKey, base: u32) -> Landings {
        let Some(g) = self.groups.get_mut(&key) else {
            return Landings::owned(Some(key), Vec::new());
        };
        note_work(g.logs.len());
        for l in &mut g.logs {
            l.copied = l.log.borrow().entries.len();
        }
        let top = g.top().expect("incr-core: a group holds a note");
        match g.log(rt) {
            Some(l) => Landings::shared(Some(key), l.log.clone(), epoch(), base, top),
            None => Landings::owned(Some(key), Vec::new()),
        }
    }

    fn remove_group(&mut self, key: PayloadKey) -> Option<Group> {
        let g = self.groups.remove(&key)?;
        self.keys.remove(&g.id);
        note_work(1);
        self.live -= g.live;
        self.renote_gen += 1;
        self.retop(key, g.top(), None);
        self.removed();
        Some(g)
    }

    /// Takes `key`'s notes, returning `rt`'s landings of them at their
    /// depths above `base`. When they are all `rt`'s, no ticket among
    /// them, and none below `base`, the group is parked and the copy
    /// shares its log, so a delivery of the copy back at `base` puts the
    /// group back instead of noting each landing again.
    pub(crate) fn take(&mut self, rt: RuntimeId, key: PayloadKey, base: u32) -> Landings {
        let epoch = epoch();
        let Some(mut g) = self.remove_group(key) else {
            return Landings::owned(Some(key), Vec::new());
        };
        let whole = g.logs.len() == 1 && g.logs[0].rt == rt && !g.lent;
        if whole && g.levels.first_key_value().is_some_and(|(&c, _)| c >= base) {
            let top = g.top().expect("incr-core: a group holds a note");
            let l = &mut g.logs[0];
            l.copied = l.log.borrow().entries.len();
            let copy = Landings::shared(Some(key), l.log.clone(), epoch, base, top);
            self.parked.push(Parked {
                group: g,
                epoch,
                base,
            });
            return copy;
        }
        let nodes = g
            .log(rt)
            .map(|l| {
                let l = l.log.borrow();
                l.landings(l.entries.len(), epoch, base, None)
            })
            .unwrap_or_default();
        drop_lent(&g, |_, _| false);
        Landings::owned(Some(key), nodes)
    }

    /// Takes `key`'s notes, returning `rt`'s restorable landings and the
    /// tickets its reads took that no copy took.
    pub(crate) fn restore(
        &mut self,
        rt: RuntimeId,
        key: PayloadKey,
    ) -> (Vec<Stamped>, Vec<Arc<Members>>) {
        let Some(g) = self.remove_group(key) else {
            return (Vec::new(), Vec::new());
        };
        let mine = g.logs.iter().position(|l| l.rt == rt);
        let restorable = |log: usize, i: usize, e: &Noted| {
            Some(log) == mine && e.gone == u64::MAX && e.restorable && i >= g.logs[log].copied
        };
        let mut lent = Vec::new();
        drop_lent(&g, |log, i| {
            let l = g.logs[log].log.borrow();
            let e = &l.entries[i];
            let keep = restorable(log, i, e);
            if keep {
                lent.extend(e.lent.as_ref().map(|(m, _)| m.clone()));
            }
            keep
        });
        let Some(l) = mine.map(|i| &g.logs[i]) else {
            return (Vec::new(), lent);
        };
        let log = l.log.borrow();
        note_work(log.entries.len());
        let nodes = newest(
            log.entries
                .iter()
                .enumerate()
                .filter(|&(i, e)| e.lent.is_none() && restorable(mine.unwrap(), i, e))
                .map(|(_, e)| (e.node, e.at)),
        );
        (nodes, lent)
    }
}

/// What a removal took of a payload's notes.
#[derive(Default)]
struct Taken {
    nodes: Vec<(u32, Stamped)>,
    lent: Vec<Arc<Members>>,
}

/// Each landing of `noted` once, in the order first noted, with the newest
/// stamp it was noted with.
fn newest(noted: impl Iterator<Item = Stamped>) -> Vec<Stamped> {
    let mut nodes: Vec<Stamped> = Vec::new();
    let mut at: HashMap<(u32, u32), usize, FxBuildHasher> = HashMap::default();
    for (node, stamp) in noted {
        match at.entry(node) {
            Entry::Occupied(i) => {
                let s = &mut nodes[*i.get()].1;
                *s = (*s).max(stamp);
            }
            Entry::Vacant(v) => {
                v.insert(nodes.len());
                nodes.push((node, stamp));
            }
        }
    }
    nodes
}

/// Drops the notes of the tickets reads took in `g`, which went with
/// their payload's notes, but those `keep` says a hold takes again.
fn drop_lent(g: &Group, mut keep: impl FnMut(usize, usize) -> bool) {
    for (log, l) in g.logs.iter().enumerate() {
        let lent: Vec<(usize, Arc<Members>)> = l
            .log
            .borrow()
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.gone == u64::MAX)
            .filter_map(|(i, e)| e.lent.as_ref().map(|(m, _)| (i, m.clone())))
            .collect();
        for (i, members) in lent {
            if !keep(log, i) {
                members.drop_note();
            }
        }
    }
}
