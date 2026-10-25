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
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering::SeqCst};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::fx::FxBuildHasher;
use crate::handle::RuntimeId;
use crate::replay::{note_work, Landings, PayloadKey};

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
    pub(crate) marked: (u64, usize),
}

#[derive(Clone, Copy)]
pub(crate) struct Member {
    pub(crate) node: (u32, u32),
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
    /// The copied set's generation and how far into `entries` it has
    /// marked the notes copies took (`replay::mark_copied`).
    pub(crate) marked: (u64, usize),
    /// As `marked`, for the kept reads' copies (`replay::keep`).
    pub(crate) kept_marked: (u64, usize),
    /// The copy of this log [`Notes::note_copy`] noted last.
    renoted: Option<Renoted>,
}

#[derive(Clone)]
pub(crate) struct Noted {
    pub(crate) node: (u32, u32),
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

impl Log {
    /// The landings of the first `len` notes still noted at removal
    /// `epoch`, each at its shallowest depth above `base`. A note below
    /// `base` counts as just above it.
    pub(crate) fn landings(&self, len: usize, epoch: u64, base: u32) -> Vec<((u32, u32), u32)> {
        self.landings_in(0, len, epoch, base)
    }

    /// As [`Self::landings`], of the notes from entry `from` on only.
    fn landings_in(
        &self,
        from: usize,
        len: usize,
        epoch: u64,
        base: u32,
    ) -> Vec<((u32, u32), u32)> {
        note_work(len - from);
        let mut nodes: Vec<((u32, u32), u32)> = Vec::new();
        let mut at: HashMap<(u32, u32), usize, FxBuildHasher> = HashMap::default();
        let mut add = |node: (u32, u32), depth: u32| match at.entry(node) {
            Entry::Occupied(i) => {
                let d = &mut nodes[*i.get()].1;
                *d = (*d).min(depth);
            }
            Entry::Vacant(v) => {
                v.insert(nodes.len());
                nodes.push((node, depth));
            }
        };
        for e in self.entries[from..len].iter().filter(|e| e.gone >= epoch) {
            let depth = e.cidx.saturating_sub(base);
            match &e.lent {
                None => add(e.node, depth),
                Some((members, len)) => {
                    note_work(*len);
                    for m in members.lock().members[..*len]
                        .iter()
                        .filter(|m| m.gone >= epoch)
                    {
                        add(m.node, depth);
                    }
                }
            }
        }
        nodes
    }
}

/// A note's place: entry `idx` of log `log` of payload `key`'s group.
#[derive(Clone, Copy)]
struct At {
    key: PayloadKey,
    log: u32,
    idx: u32,
}

/// The notes of one payload.
struct Group {
    logs: Vec<RuntimeLog>,
    /// The positions with a note, with the notes there, some of which may
    /// have gone since.
    levels: BTreeMap<u32, Level>,
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
}

struct Renoted {
    renote_gen: u64,
    len: usize,
    epoch: u64,
    from_base: u32,
    rt: RuntimeId,
    key: PayloadKey,
    base: u32,
}

impl Notes {
    pub(crate) const fn new() -> Self {
        Self {
            groups: BTreeMap::new(),
            tops: BTreeSet::new(),
            by_node: BTreeMap::new(),
            live: 0,
            renote_gen: 0,
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
        node: (u32, u32),
        restorable: bool,
    ) {
        self.push_noted(rt, key, cidx, node, None, restorable);
    }

    /// Notes, under `key` and above `base`, the landings a copy holds: the
    /// first `len` entries of `log` still noted at removal `epoch`, at
    /// their depths above `from_base`. When the copy of `log` noted last
    /// went to the same place, at an epoch no later, and no note has gone
    /// since, the entries it held are noted already, so only later ones
    /// are: a second note of a node at the same place, or deeper than
    /// another, retires with that one and adds no landing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn note_copy(
        &mut self,
        rt: RuntimeId,
        key: PayloadKey,
        base: u32,
        log: &Rc<RefCell<Log>>,
        len: usize,
        epoch: u64,
        from_base: u32,
    ) {
        let from = match &log.borrow().renoted {
            Some(r)
                if (r.renote_gen, r.rt, r.key, r.base, r.from_base)
                    == (self.renote_gen, rt, key, base, from_base)
                    && r.len <= len
                    && r.epoch <= epoch =>
            {
                r.len
            }
            _ => 0,
        };
        let nodes = log.borrow().landings_in(from, len, epoch, from_base);
        for (node, depth) in nodes {
            self.push(rt, key, base + depth, node, false);
        }
        log.borrow_mut().renoted = Some(Renoted {
            renote_gen: self.renote_gen,
            len,
            epoch,
            from_base,
            rt,
            key,
            base,
        });
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
        self.push_noted(rt, key, cidx, (u32::MAX, 0), Some((members, len)), true);
    }

    fn push_noted(
        &mut self,
        rt: RuntimeId,
        key: PayloadKey,
        cidx: u32,
        node: (u32, u32),
        lent: Option<(Arc<Members>, usize)>,
        restorable: bool,
    ) {
        note_work(1);
        let indexed = lent.is_none();
        let g = self.groups.entry(key).or_insert_with(|| Group {
            logs: Vec::new(),
            levels: BTreeMap::new(),
        });
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
        let new = g.top();
        self.retop(key, old, new);
        if indexed {
            self.by_node
                .entry((rt, node))
                .or_default()
                .push(At { key, log, idx });
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
        if self.live == 0 {
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
            let Some(g) = self.groups.get_mut(&at.key) else {
                continue;
            };
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
            if level.live == 0 {
                g.levels.remove(&cidx);
            }
            let new = g.top();
            if new.is_none() {
                self.groups.remove(&at.key);
            }
            self.live -= 1;
            self.renote_gen += 1;
            self.retop(at.key, old, new);
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
            self.renote_gen += 1;
            for (log, idx) in level.at {
                let Some(e) = g.take(log, idx, removal) else {
                    continue;
                };
                match (e.lent, mine == Some(log as usize)) {
                    (None, true) => out.nodes.push((idx, e.node)),
                    (None, false) => {}
                    (Some((members, _)), true) => out.lent.push(members),
                    (Some((members, _)), false) => members.drop_note(),
                }
            }
        }
        let new = g.top();
        if new.is_none() {
            self.groups.remove(&key);
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
    ) -> (Vec<(u32, u32)>, Vec<Arc<Members>>) {
        let mut ours = self.forget_levels(key, cidx, Some(rt));
        if let Some(above) = cidx.checked_add(1) {
            self.forget_from(above, None);
        } else {
            self.removed();
        }
        ours.nodes.sort_unstable_by_key(|&(idx, _)| idx);
        let mut seen = BTreeSet::new();
        let nodes = ours
            .nodes
            .into_iter()
            .filter_map(|(_, node)| seen.insert(node).then_some(node))
            .collect();
        (nodes, ours.lent)
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
        let live: u32 = g.levels.values().map(|l| l.live).sum();
        note_work(g.levels.len());
        self.live -= live as usize;
        self.renote_gen += 1;
        self.retop(key, g.top(), None);
        self.removed();
        Some(g)
    }

    /// Takes `key`'s notes, returning `rt`'s landings of them at their
    /// depths above `base`.
    pub(crate) fn take(&mut self, rt: RuntimeId, key: PayloadKey, base: u32) -> Landings {
        let epoch = epoch();
        let Some(g) = self.remove_group(key) else {
            return Landings::owned(Some(key), Vec::new());
        };
        let nodes = g
            .log(rt)
            .map(|l| {
                let l = l.log.borrow();
                l.landings(l.entries.len(), epoch, base)
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
    ) -> (Vec<(u32, u32)>, Vec<Arc<Members>>) {
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
        let mut seen = BTreeSet::new();
        let nodes = log
            .entries
            .iter()
            .enumerate()
            .filter(|&(i, e)| e.lent.is_none() && restorable(mine.unwrap(), i, e))
            .filter_map(|(_, e)| seen.insert(e.node).then_some(e.node))
            .collect();
        (nodes, lent)
    }
}

/// What a removal took of a payload's notes.
#[derive(Default)]
struct Taken {
    nodes: Vec<(u32, (u32, u32))>,
    lent: Vec<Arc<Members>>,
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
