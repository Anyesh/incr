//! Doom reservations under `Shared`.
//!
//! A doom releases claims whose nodes the dooming thread's replay contexts
//! compute again, where the unbounded run keeps them claimed on that
//! thread's stack. A reservation keeps every other thread from computing
//! such a node until the thread lands it settled from a finished run,
//! where the unbounded run's frame releases its claim, or until the
//! thread's outermost context of that runtime, which outlives every
//! context nested in it, is removed; the dooming thread's own claims and unrun releases
//! of the node keep it. Once the thread's frame of the node is left for a
//! retry that reruns from the closure start, the reservation rests: it
//! holds only ops begun before then, since a later op may be a reader the
//! retry itself started. An op begun after the rest can still compute the
//! node ahead of the retry. A reservation is stored
//! before the release that publishes it, so a claim that reads the release
//! sees it. A node generation has at most one reservation: reserving it
//! needs its claim, and a claim waits out another thread's reservation.
//!
//! A read's walk that lands the read's own node Dirty reserves it the same
//! way until the read has taken the value: another thread's rerun of the
//! Dirty node would replace it with a value no serial order gives the read.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashSet};
#[cfg(not(loom))]
use std::hash::{Hash, Hasher};

use crate::fx::FxBuildHasher;
#[cfg(not(loom))]
use crate::fx::FxHasher;
use crate::waits::Claim;

#[cfg(not(loom))]
use std::sync::{
    atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    Mutex,
};

#[cfg(loom)]
use loom::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Mutex,
};

/// Slots of [`BUCKETS`]; a power of two.
#[cfg(not(loom))]
const BUCKETS_LEN: usize = 1024;

/// A reservation's node generation, its owner's token, and the [`SEQ`]
/// value its owner left its run at ([`AWAKE`] while every op waits on it).
type Entry = (u32, usize, u64);

/// The stamp of a reservation every other reader waits on.
const AWAKE: u64 = u64::MAX;

/// Entries in [`TABLE`], and per hash bucket of their claims, both stored
/// under its lock: a claim that reads a release sees non-zero counts
/// without taking the lock, and takes it only on its own bucket's hit.
#[cfg(not(loom))]
static LIVE: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(loom))]
static BUCKETS: [AtomicU32; BUCKETS_LEN] = [const { AtomicU32::new(0) }; BUCKETS_LEN];

#[cfg(not(loom))]
static TABLE: Mutex<BTreeMap<Claim, Entry>> = Mutex::new(BTreeMap::new());

/// Orders the start of each op that leaves the Clean fast path against
/// [`rest`]s.
#[cfg(not(loom))]
static SEQ: AtomicU64 = AtomicU64::new(1);

#[cfg(loom)]
loom::lazy_static! {
    static ref LIVE: AtomicUsize = AtomicUsize::new(0);
    static ref TABLE: Mutex<BTreeMap<Claim, Entry>> = Mutex::new(BTreeMap::new());
    static ref SEQ: AtomicU64 = AtomicU64::new(1);
}

/// One context's reservations on this thread, as claim and generation.
struct Group {
    context: u64,
    claims: Claims,
}

/// The claims and generations a group lists: a set, so a landing ends
/// one without a pass over a deep chain's reservations. No group lists
/// one twice: `note` lists again only for a context with none, and
/// `note_root` only when the read-root group has none.
#[derive(Default)]
struct Claims(HashSet<(Claim, u32), FxBuildHasher>);

impl Claims {
    fn one(c: (Claim, u32)) -> Self {
        let mut claims = Self::default();
        claims.push(c);
        claims
    }

    fn push(&mut self, c: (Claim, u32)) {
        self.0.insert(c);
    }

    fn contains(&self, c: &(Claim, u32)) -> bool {
        self.0.contains(c)
    }

    fn remove(&mut self, c: &(Claim, u32)) -> bool {
        self.0.remove(c)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[cfg(all(test, not(loom)))]
    fn len(&self) -> u32 {
        self.0.len() as u32
    }

    fn extend(&mut self, other: Claims) {
        self.0.extend(other.0);
    }

    fn keys(&self) -> impl Iterator<Item = (Claim, u32)> + '_ {
        self.0.iter().copied()
    }
}

/// A thread's reservations. Its address is the owner token, unique among
/// live threads; its drop at thread exit ends any its contexts did not,
/// since the contexts' own thread-local may be destroyed after it.
struct Listed(RefCell<Vec<Group>>);

impl Listed {
    fn token(&self) -> usize {
        std::ptr::from_ref(self) as usize
    }
}

impl Drop for Listed {
    fn drop(&mut self) {
        let token = self.token();
        for g in std::mem::take(self.0.get_mut()) {
            unlist(g.claims.keys(), token, false);
        }
    }
}

#[cfg(not(loom))]
thread_local! {
    static MINE: Listed = const { Listed(RefCell::new(Vec::new())) };
    static SCOPE: Cell<Option<u64>> = const { Cell::new(None) };
    /// The [`root_key`] of the innermost read's node, 0 outside reads.
    static ROOT: Cell<u64> = const { Cell::new(0) };
    /// Claims the read-root group lists, so a read's end skips it empty.
    static ROOTS: Cell<u32> = const { Cell::new(0) };
    /// [`SEQ`] as this thread's current op left the Clean fast path; 0
    /// before its first.
    static OP: Cell<u64> = const { Cell::new(0) };
}

#[cfg(loom)]
loom::thread_local! {
    static MINE: Listed = Listed(RefCell::new(Vec::new()));
    static SCOPE: Cell<Option<u64>> = Cell::new(None);
    static ROOT: Cell<u64> = Cell::new(0);
    static ROOTS: Cell<u32> = Cell::new(0);
    static OP: Cell<u64> = Cell::new(0);
}

/// The context of a doom's reservations until the doom has marked one.
pub(crate) const PENDING: u64 = 0;

/// The context of a read's reservation of its own node, which no replay
/// context's end ends.
const ROOT_READ: u64 = u64::MAX;

/// The context bit of a read's walk group (`WALK | root_key`), which holds
/// what the walk lands unsettled until the read ends.
const WALK: u64 = 1 << 63;

/// Whether a group of `context` lasts as long as a read, not a context.
fn for_read(context: u64) -> bool {
    context == ROOT_READ || context & WALK != 0
}

fn table() -> impl std::ops::DerefMut<Target = BTreeMap<Claim, Entry>> {
    #[cfg(not(loom))]
    {
        TABLE.lock().unwrap_or_else(|e| e.into_inner())
    }
    #[cfg(loom)]
    {
        TABLE.lock().unwrap()
    }
}

#[cfg(not(loom))]
fn bucket(claim: Claim) -> &'static AtomicU32 {
    let mut h = FxHasher::default();
    claim.hash(&mut h);
    &BUCKETS[h.finish() as usize & (BUCKETS_LEN - 1)]
}

/// Whether `claim`'s bucket counts any reservation.
#[cfg(not(loom))]
fn bucket_hit(claim: Claim) -> bool {
    bucket(claim).load(Ordering::Relaxed) != 0
}

/// Loom builds keep no bucket counts: loom's modification-order check
/// (`assert_ne!(mo_i, mo_j)` in rt/atomic.rs, marked "sometimes fails")
/// trips on them. [`LIVE`], stored and read the same way, still gates the
/// table lock.
#[cfg(loom)]
fn bucket_hit(_: Claim) -> bool {
    true
}

/// While it lives, every claim this thread releases unrun is reserved for
/// `context`, if any.
pub(crate) struct Scope(Option<u64>);

impl Scope {
    pub(crate) fn new(context: Option<u64>) -> Self {
        Self(SCOPE.with(|s| s.replace(context)))
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        SCOPE.with(|s| s.set(self.0));
    }
}

/// Reserves `claim` at `generation` for the scope's context, if a
/// [`Scope`] is open and no context of this thread reserves it already.
/// Called before the release. A claim only a read group lists is listed
/// again, awake: the one way a walk's listing at rest is woken.
pub(crate) fn note(claim: Claim, generation: u32) {
    let Some(context) = SCOPE.with(Cell::get) else {
        return;
    };
    // Only the read-root group may list it besides this one; a context
    // that reserves it already keeps it, as without a read.
    let again = ROOTS.with(Cell::get) != 0
        && MINE.with(|m| {
            let groups = m.0.borrow();
            let by = |root: bool| {
                groups
                    .iter()
                    .any(|g| for_read(g.context) == root && g.claims.contains(&(claim, generation)))
            };
            by(true) && !by(false)
        });
    list(context, claim, generation, again);
}

/// `claim` as [`ROOT`] holds it. Runtime ids are allocated from 1 and
/// stay far below 2^32, so distinct claims never share a key, and no
/// claim's key is 0.
#[inline(always)]
fn root_key((rt, slot): Claim) -> u64 {
    (rt.get() << 32) | u64::from(slot)
}

/// While it lives, this thread's read of `claim` is walking or taking the
/// value its walk left: a Dirty landing of `claim` reserves it until the
/// guard drops.
pub(crate) struct Root {
    key: u64,
    outer: u64,
}

impl Root {
    /// `on` is false under `Local`, where the guard does nothing.
    #[inline(always)]
    pub(crate) fn arm(on: bool, claim: Claim) -> Self {
        if !on {
            return Self { key: 0, outer: 0 };
        }
        let key = root_key(claim);
        let outer = ROOT.with(|r| r.replace(key));
        Self { key, outer }
    }
}

impl Drop for Root {
    #[inline(always)]
    fn drop(&mut self) {
        // An enclosing read of the same node, or none under `Local`, takes
        // the value after this one, so the reservation passes to it.
        if self.outer != self.key {
            ROOT.with(|r| r.set(self.outer));
            if ROOTS.with(Cell::get) != 0 {
                end_root(self.key);
            }
        }
    }
}

/// Reserves `claim`, landing Dirty, for the read whose node it is, if
/// this thread's innermost read is that read. Called before the release.
#[cold]
#[inline(never)]
pub(crate) fn note_root(claim: Claim, generation: u32) {
    if ROOT.with(Cell::get) != root_key(claim) || listed(ROOT_READ, claim, generation) {
        return;
    }
    // Listed even if a context of this thread reserves it already, so it
    // stays reserved until both have ended.
    list(ROOT_READ, claim, generation, true);
    ROOTS.with(|n| n.set(n.get() + 1));
}

/// Reserves `claim`, which a run inside this thread's innermost read just
/// landed New or Dirty in a budgeted runtime, until that read ends, at
/// rest from now. A walk entry's contained unwind lands claims its walk
/// runs again outside any doom's scope, and the context whose end follows
/// the containment no longer holds them. Ops begun before wait, as the unbounded run's claim would
/// make them; later ones, which the read's own retry may spawn and join,
/// pass. The read's own node is [`note_root`]'s.
#[cold]
#[inline(never)]
pub(crate) fn note_walk(claim: Claim, generation: u32) {
    let key = ROOT.with(Cell::get);
    if key == 0 || key == root_key(claim) {
        return;
    }
    let context = WALK | key;
    if listed(context, claim, generation) {
        return;
    }
    let listed = MINE.with(|m| {
        let token = m.token();
        {
            let mut t = table();
            match t.get(&claim) {
                // Another thread's reservation stands for the node; at
                // rest it holds fewer ops than this listing would, the known
                // later-op gap.
                Some(&(g, owner, _)) if g == generation && owner != token => return false,
                Some(&(g, _, _)) if g == generation => {}
                _ => {
                    let stamp = SEQ.fetch_add(1, Ordering::SeqCst);
                    if t.insert(claim, (generation, token, stamp)).is_none() {
                        #[cfg(not(loom))]
                        bucket(claim).fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            LIVE.store(t.len(), Ordering::Relaxed);
        }
        let mut groups = m.0.borrow_mut();
        match groups.iter_mut().find(|g| g.context == context) {
            Some(g) => g.claims.push((claim, generation)),
            None => groups.push(Group {
                context,
                claims: Claims::one((claim, generation)),
            }),
        }
        true
    });
    if listed {
        ROOTS.with(|n| n.set(n.get() + 1));
    }
}

/// Whether this thread's group of `context` lists `claim`.
fn listed(context: u64, claim: Claim, generation: u32) -> bool {
    MINE.with(|m| {
        m.0.borrow()
            .iter()
            .any(|g| g.context == context && g.claims.contains(&(claim, generation)))
    })
}

#[cold]
#[inline(never)]
fn end_root(key: u64) {
    let gone = MINE
        .try_with(|m| {
            let mut groups = m.0.borrow_mut();
            let mut ended: Vec<_> = groups
                .iter()
                .position(|g| g.context == (WALK | key))
                .map(|at| groups.swap_remove(at).claims.keys().collect())
                .unwrap_or_default();
            if let Some(g) = groups.iter_mut().find(|g| g.context == ROOT_READ) {
                ended.extend(g.claims.0.extract_if(|&(c, _)| root_key(c) == key));
                if g.claims.is_empty() {
                    groups.retain(|g| g.context != ROOT_READ);
                }
            }
            ROOTS.with(|n| n.set(n.get() - ended.len() as u32));
            Some((ended, m.token()))
        })
        .ok()
        .flatten();
    if let Some((ended, token)) = gone {
        unlist(ended, token, true);
    }
}

fn list(context: u64, claim: Claim, generation: u32, again: bool) {
    MINE.with(|m| {
        let token = m.token();
        {
            let mut t = table();
            match t.insert(claim, (generation, token, AWAKE)) {
                Some((g, owner, _)) if (g, owner) == (generation, token) && !again => return,
                Some(_) => {}
                None => {
                    #[cfg(not(loom))]
                    bucket(claim).fetch_add(1, Ordering::Relaxed);
                }
            }
            LIVE.store(t.len(), Ordering::Relaxed);
        }
        let mut groups = m.0.borrow_mut();
        match groups.iter_mut().find(|g| g.context == context) {
            Some(g) => g.claims.push((claim, generation)),
            None => groups.push(Group {
                context,
                claims: Claims::one((claim, generation)),
            }),
        }
    });
}

/// Whether any thread holds a reservation. Read after a claim CAS, which
/// orders it after the release that published a reservation of the node.
#[inline(always)]
pub(crate) fn any() -> bool {
    LIVE.load(Ordering::Relaxed) != 0
}

/// Whether `claim` is reserved at `generation`, and if so whether by this
/// thread and the reservation's stamp.
fn reserved(claim: Claim, generation: u32) -> Option<(bool, u64)> {
    if !any() || !bucket_hit(claim) {
        return None;
    }
    let token = MINE.with(Listed::token);
    table()
        .get(&claim)
        .filter(|&&(g, _, _)| g == generation)
        .map(|&(_, owner, stamp)| (owner == token, stamp))
}

/// Whether another thread reserved `claim` at `generation` against this
/// thread's op. A reservation at rest holds only ops begun before its
/// owner left the node's run: a later op is a reader the owner's retry
/// may itself have started, before that retry reads the node again.
#[cold]
#[inline(never)]
pub(crate) fn elsewhere(claim: Claim, generation: u32) -> bool {
    matches!(reserved(claim, generation), Some((false, stamp)) if OP.with(Cell::get) <= stamp)
}

/// Whether this thread reserved `claim` at `generation`.
#[cold]
#[inline(never)]
pub(crate) fn mine(claim: Claim, generation: u32) -> bool {
    matches!(reserved(claim, generation), Some((true, _)))
}

/// Starts this thread's op at the current [`SEQ`].
#[inline]
pub(crate) fn op_begin() {
    OP.with(|o| o.set(SEQ.load(Ordering::SeqCst)));
}

/// Leaves this thread's reservation of `claim` at `generation` at rest:
/// its owner's retry reads the node afresh once it has run what precedes
/// that read again, so only ops begun before now still wait on it. A
/// reservation already at rest keeps its stamp, so the readers an earlier
/// retry started still pass; a read's reservation of its own node stays
/// awake until the read has taken the value.
#[cold]
#[inline(never)]
pub(crate) fn rest(claim: Claim, generation: u32) {
    let Ok((token, root)) = MINE.try_with(|m| {
        let root =
            m.0.borrow()
                .iter()
                .any(|g| g.context == ROOT_READ && g.claims.contains(&(claim, generation)));
        (m.token(), root)
    }) else {
        return;
    };
    if root {
        return;
    }
    if let Some(e) = table().get_mut(&claim) {
        if (e.0, e.1) == (generation, token) && e.2 == AWAKE {
            e.2 = SEQ.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// Ends `claims`, already out of their group. `shared` says another
/// group may list them too, which only a read's reservation of its own
/// node makes possible.
fn unlist(claims: impl IntoIterator<Item = (Claim, u32)>, token: usize, shared: bool) {
    let mut t = table();
    for (claim, generation) in claims {
        let Some(e) = t.get_mut(&claim) else {
            continue;
        };
        if (e.0, e.1) != (generation, token) {
            continue;
        }
        match shared.then(|| still_listed_by(claim, generation)).flatten() {
            // Only a walk group keeps it: at rest from here, as listed.
            Some(Lister::Walk) => {
                if e.2 == AWAKE {
                    e.2 = SEQ.fetch_add(1, Ordering::SeqCst);
                }
            }
            Some(Lister::Other) => {}
            None => {
                t.remove(&claim);
                #[cfg(not(loom))]
                bucket(claim).fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
    LIVE.store(t.len(), Ordering::Relaxed);
}

/// Which of this thread's groups still list a claim.
enum Lister {
    /// Walk groups only.
    Walk,
    /// A context or the read-root group.
    Other,
}

/// Whether this thread's read-root group may list anything; true once
/// the count is being destroyed.
fn roots_listed() -> bool {
    ROOTS.try_with(Cell::get).ok().is_none_or(|n| n != 0)
}

/// Which groups of this thread still list `claim`, as a read of its own
/// node or its walk and a context can both. None once the thread's
/// reservations are being destroyed.
fn still_listed_by(claim: Claim, generation: u32) -> Option<Lister> {
    MINE.try_with(|m| {
        let groups = m.0.borrow();
        let mut by = groups
            .iter()
            .filter(|g| g.claims.contains(&(claim, generation)));
        let first = by.next()?;
        let walk = |c: u64| c & WALK != 0 && c != ROOT_READ;
        Some(if walk(first.context) && by.all(|g| walk(g.context)) {
            Lister::Walk
        } else {
            Lister::Other
        })
    })
    .ok()
    .flatten()
}

/// Whether a context of this thread reserves `claim` at `generation`.
pub(crate) fn in_context(claim: Claim, generation: u32) -> bool {
    MINE.try_with(|m| {
        m.0.borrow()
            .iter()
            .any(|g| !for_read(g.context) && g.claims.contains(&(claim, generation)))
    })
    .unwrap_or(false)
}

/// Ends this thread's context reservations of `claim` at `generation`,
/// whose run has landed for good. A read's reservation of its own node
/// stays until the read has taken the value.
pub(crate) fn end_landed(claim: Claim, generation: u32) {
    let ended = MINE.try_with(|m| {
        let mut groups = m.0.borrow_mut();
        let mut found = false;
        for g in groups.iter_mut().filter(|g| !for_read(g.context)) {
            found |= g.claims.remove(&(claim, generation));
        }
        groups.retain(|g| !g.claims.is_empty() || for_read(g.context));
        found.then(|| m.token())
    });
    if let Ok(Some(token)) = ended {
        unlist([(claim, generation)], token, roots_listed());
    }
}

/// Gives the reservations of a doom that marked no context to `context`.
pub(crate) fn adopt(context: u64) {
    MINE.with(|m| {
        let mut groups = m.0.borrow_mut();
        let Some(at) = groups.iter().position(|g| g.context == PENDING) else {
            return;
        };
        let pending = groups.swap_remove(at);
        match groups.iter_mut().find(|g| g.context == context) {
            Some(g) => g.claims.extend(pending.claims),
            None => groups.push(Group {
                context,
                claims: pending.claims,
            }),
        }
    });
}

#[cfg(all(test, not(loom)))]
thread_local! {
    static CONTEXT_ENDS: Cell<u32> = const { Cell::new(0) };
}

/// Reservations this thread's context removals ended, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) fn context_ends() -> u32 {
    CONTEXT_ENDS.with(Cell::get)
}

/// Ends the reservations of `context`, which is being removed.
pub(crate) fn end_context(context: u64) {
    let gone = MINE
        .try_with(|m| {
            let mut groups = m.0.borrow_mut();
            let at = groups.iter().position(|g| g.context == context)?;
            Some((groups.swap_remove(at), m.token()))
        })
        .ok()
        .flatten();
    if let Some((g, token)) = gone {
        #[cfg(all(test, not(loom)))]
        CONTEXT_ENDS.with(|n| n.set(n.get() + g.claims.len()));
        unlist(g.claims.keys(), token, roots_listed());
    }
}

/// Ends every reservation of this thread, PENDING ones and a read's of
/// its own node included: none of its contexts outlives this.
pub(crate) fn end_all() {
    ROOTS.with(|n| n.set(0));
    let (groups, token) = MINE.with(|m| (std::mem::take(&mut *m.0.borrow_mut()), m.token()));
    for g in groups {
        unlist(g.claims.keys(), token, false);
    }
}

/// [`end_all`], except a read's reservation of its own node, which the
/// read ends once it has taken the value: a context's removal, even at the
/// outermost entry's exit, comes before that.
pub(crate) fn end_contexts() {
    let (groups, token) = MINE.with(|m| {
        let mut groups = m.0.borrow_mut();
        let ended = groups
            .extract_if(.., |g| !for_read(g.context))
            .collect::<Vec<_>>();
        (ended, m.token())
    });
    for g in groups {
        unlist(g.claims.keys(), token, roots_listed());
    }
}

/// The claims this thread has reserved, which its claim waits list as
/// held.
pub(crate) fn held() -> Vec<Claim> {
    MINE.with(|m| {
        m.0.borrow()
            .iter()
            .flat_map(|g| g.claims.keys().map(|(c, _)| c))
            .collect()
    })
}

/// Whether this thread holds no reservation, as it must once its
/// contexts are gone.
pub(crate) fn none_held() -> bool {
    MINE.with(|m| m.0.borrow().is_empty())
}

/// Ends a context's reservations when the context is dropped.
pub(crate) struct Ends(pub(crate) u64);

impl Drop for Ends {
    fn drop(&mut self) {
        end_context(self.0);
    }
}
