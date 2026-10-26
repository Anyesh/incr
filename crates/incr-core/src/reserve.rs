//! Doom reservations under `Shared`.
//!
//! A doom releases claims whose nodes the dooming thread's replay contexts
//! compute again, where the unbounded run keeps them claimed on that
//! thread's stack. A reservation keeps every other thread from computing
//! such a node until the thread lands it settled from a finished run,
//! where the unbounded run's frame releases its claim, or until the
//! thread's outermost context of that runtime, which outlives every
//! context nested in it, is removed; the dooming thread's own claims and unrun releases
//! of the node keep it. It is stored
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

use crate::attempt::{self, Attempt};
use crate::fx::FxBuildHasher;
#[cfg(not(loom))]
use crate::fx::FxHasher;
use crate::waits::Claim;

#[cfg(not(loom))]
use std::sync::{
    atomic::{AtomicU32, AtomicUsize, Ordering},
    Mutex,
};

#[cfg(loom)]
use loom::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

/// Slots of [`BUCKETS`]; a power of two.
#[cfg(not(loom))]
const BUCKETS_LEN: usize = 1024;

/// A reservation's node generation and its owner's token.
type Entry = (u32, usize);

/// Entries in [`TABLE`], and per hash bucket of their claims, both stored
/// under its lock: a claim that reads a release sees non-zero counts
/// without taking the lock, and takes it only on its own bucket's hit.
#[cfg(not(loom))]
static LIVE: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(loom))]
static BUCKETS: [AtomicU32; BUCKETS_LEN] = [const { AtomicU32::new(0) }; BUCKETS_LEN];

#[cfg(not(loom))]
static TABLE: Mutex<BTreeMap<Claim, (Entry, Attempt)>> = Mutex::new(BTreeMap::new());

#[cfg(loom)]
loom::lazy_static! {
    static ref LIVE: AtomicUsize = AtomicUsize::new(0);
    static ref TABLE: Mutex<BTreeMap<Claim, (Entry, Attempt)>> = Mutex::new(BTreeMap::new());
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
    /// Claims waiters met as they took over this thread's attempts, which
    /// never doom again before its outermost entry exits.
    static TAKEN: RefCell<Vec<Claim>> = const { RefCell::new(Vec::new()) };
    /// Taken claims this thread landed for a Cyclic outcome, at their
    /// generations, not read since: published Dirty, with no value.
    static TAKEN_CYCLIC: RefCell<Vec<(Claim, u32)>> = const { RefCell::new(Vec::new()) };
    static MINE: Listed = const { Listed(RefCell::new(Vec::new())) };
    static SCOPE: Cell<Option<u64>> = const { Cell::new(None) };
    /// The [`root_key`] of the innermost read's node, 0 outside reads.
    static ROOT: Cell<u64> = const { Cell::new(0) };
    /// Claims the read-root group lists, so a read's end skips it empty.
    static ROOTS: Cell<u32> = const { Cell::new(0) };
}

#[cfg(loom)]
loom::thread_local! {
    static TAKEN: RefCell<Vec<Claim>> = RefCell::new(Vec::new());
    static TAKEN_CYCLIC: RefCell<Vec<(Claim, u32)>> = RefCell::new(Vec::new());
    static MINE: Listed = Listed(RefCell::new(Vec::new()));
    static SCOPE: Cell<Option<u64>> = Cell::new(None);
    static ROOT: Cell<u64> = Cell::new(0);
    static ROOTS: Cell<u32> = Cell::new(0);
}

/// The context of a doom's reservations until the doom has marked one.
pub(crate) const PENDING: u64 = 0;

/// The context of a read's reservation of its own node, which no replay
/// context's end ends.
const ROOT_READ: u64 = u64::MAX;

fn table() -> impl std::ops::DerefMut<Target = BTreeMap<Claim, (Entry, Attempt)>> {
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
/// Called before the release.
pub(crate) fn note(claim: Claim, generation: u32) {
    let Some(context) = SCOPE.with(Cell::get) else {
        return;
    };
    if taken(claim) {
        return;
    }
    // Only the read-root group may list it besides this one; a context
    // that reserves it already keeps it, as without a read.
    let again = ROOTS.with(Cell::get) != 0
        && MINE.with(|m| {
            let groups = m.0.borrow();
            let by = |root: bool| {
                groups.iter().any(|g| {
                    (g.context == ROOT_READ) == root && g.claims.contains(&(claim, generation))
                })
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
    if ROOT.with(Cell::get) != root_key(claim)
        || listed(ROOT_READ, claim, generation)
        || taken(claim)
    {
        return;
    }
    // Listed even if a context of this thread reserves it already, so it
    // stays reserved until both have ended.
    list(ROOT_READ, claim, generation, true);
    ROOTS.with(|n| n.set(n.get() + 1));
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
            let g = groups.iter_mut().find(|g| g.context == ROOT_READ)?;
            let ended: Vec<_> = g
                .claims
                .0
                .extract_if(|&(c, _)| root_key(c) == key)
                .collect();
            ROOTS.with(|n| n.set(n.get() - ended.len() as u32));
            if g.claims.is_empty() {
                groups.retain(|g| g.context != ROOT_READ);
            }
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
            match t.insert(claim, ((generation, token), attempt::mine())) {
                Some((prev, _)) if prev == (generation, token) && !again => return,
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
/// thread.
fn reserved(claim: Claim, generation: u32) -> Option<bool> {
    if !any() || !bucket_hit(claim) {
        return None;
    }
    let token = MINE.with(Listed::token);
    table()
        .get(&claim)
        .filter(|((g, _), a)| *g == generation && !a.wounded())
        .map(|&((_, owner), _)| owner == token)
}

/// The attempt of another thread that reserved `claim` at `generation`.
fn owner_elsewhere(claim: Claim, generation: u32) -> Option<Attempt> {
    if !any() || !bucket_hit(claim) {
        return None;
    }
    let token = MINE.with(Listed::token);
    table()
        .get(&claim)
        .filter(|&&((g, owner), _)| g == generation && owner != token)
        .map(|(_, a)| a.clone())
}

/// Whether another thread reserved `claim` at `generation` for an attempt
/// no waiter has wounded: a wounded attempt's reservations hold no one.
#[cold]
#[inline(never)]
pub(crate) fn elsewhere(claim: Claim, generation: u32) -> bool {
    owner_elsewhere(claim, generation).is_some_and(|a| !a.wounded())
}

/// A waiter on another thread's reservation of `claim`: wounds the
/// owner's attempt once its discard has stayed in flight through the
/// waiter's `patience`, which frees the reservation. True if the
/// reservation holds no one now.
#[cold]
#[inline(never)]
pub(crate) fn take_over(claim: Claim, generation: u32, patience: &mut attempt::Patience) -> bool {
    owner_elsewhere(claim, generation).is_none_or(|a| a.wound_if_stuck(claim, patience))
}

/// Whether this thread reserved `claim` at `generation`.
#[cold]
#[inline(never)]
pub(crate) fn mine(claim: Claim, generation: u32) -> bool {
    reserved(claim, generation) == Some(true)
}

/// Ends `claims`, already out of their group. `shared` says another
/// group may list them too, which only a read's reservation of its own
/// node makes possible.
fn unlist(claims: impl IntoIterator<Item = (Claim, u32)>, token: usize, shared: bool) {
    let mut t = table();
    for (claim, generation) in claims {
        if t.get(&claim).map(|e| e.0) == Some((generation, token))
            && !(shared && still_listed(claim, generation))
        {
            t.remove(&claim);
            #[cfg(not(loom))]
            bucket(claim).fetch_sub(1, Ordering::Relaxed);
        }
    }
    LIVE.store(t.len(), Ordering::Relaxed);
}

/// Whether this thread's read-root group may list anything; true once
/// the count is being destroyed.
fn roots_listed() -> bool {
    ROOTS.try_with(Cell::get).ok().is_none_or(|n| n != 0)
}

/// Whether a group of this thread still lists `claim`, as a read of its
/// own node and a context can both. False once the thread's reservations
/// are being destroyed.
fn still_listed(claim: Claim, generation: u32) -> bool {
    MINE.try_with(|m| {
        m.0.borrow()
            .iter()
            .any(|g| g.claims.contains(&(claim, generation)))
    })
    .unwrap_or(false)
}

/// Whether a context of this thread reserves `claim` at `generation`.
pub(crate) fn in_context(claim: Claim, generation: u32) -> bool {
    MINE.try_with(|m| {
        m.0.borrow()
            .iter()
            .any(|g| g.context != ROOT_READ && g.claims.contains(&(claim, generation)))
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
        for g in groups.iter_mut().filter(|g| g.context != ROOT_READ) {
            found |= g.claims.remove(&(claim, generation));
        }
        groups.retain(|g| !g.claims.is_empty() || g.context == ROOT_READ);
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
            .extract_if(.., |g| g.context != ROOT_READ)
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

/// A waiter took over this thread's attempt: the claims its waiters met
/// are never doomed, reserved or held again before the thread's outermost
/// entry exits, except as members of an episode held once more because
/// another member carries hold state from the attempt, which marks them
/// all. Each wounded attempt so adds at least one claim it reserved or held
/// while untaken, and a read restarts for take-overs at most once per
/// claim it can reserve or hold.
pub(crate) fn mark_taken() {
    let claims = attempt::take_met();
    if claims.is_empty() {
        return;
    }
    #[cfg(all(test, not(loom)))]
    TAKES.with(|n| n.set(n.get() + 1));
    mark_claims_taken(claims);
}

/// `claims` are taken, as [`mark_taken`]'s are.
pub(crate) fn mark_claims_taken(claims: impl IntoIterator<Item = Claim>) {
    TAKEN.with(|t| t.borrow_mut().extend(claims));
    crate::cycle_state::set_any_taken(true);
}

/// Whether a take-over during the thread's outermost entry met `claim`.
#[inline(always)]
pub(crate) fn taken(claim: Claim) -> bool {
    crate::cycle_state::any_taken() && taken_slow(claim)
}

#[cold]
#[inline(never)]
fn taken_slow(claim: Claim) -> bool {
    TAKEN.with(|t| t.borrow().contains(&claim))
}

/// This thread landed taken `claim` at `generation`, for a Cyclic outcome
/// if `cyclic`: the next read of it on this thread, and only that one,
/// gives that cycle, so no later landing by any thread is read as it.
pub(crate) fn note_taken_landing(claim: Claim, generation: u32, cyclic: bool) {
    TAKEN_CYCLIC.with(|t| {
        let mut t = t.borrow_mut();
        t.retain(|&(c, _)| c != claim);
        if cyclic {
            t.push((claim, generation));
        }
    });
}

/// Whether this thread's last landing of taken `claim` at `generation`
/// was for a Cyclic outcome not read since; the read takes it.
pub(crate) fn take_taken_cyclic(claim: Claim, generation: u32) -> bool {
    crate::cycle_state::any_taken()
        && TAKEN_CYCLIC.with(|t| {
            let mut t = t.borrow_mut();
            let at = t.iter().position(|&n| n == (claim, generation));
            at.map(|at| t.swap_remove(at)).is_some()
        })
}

#[cfg(all(test, not(loom)))]
thread_local! {
    static TAKES: Cell<u32> = const { Cell::new(0) };
}

/// Take-overs during the thread's outermost entry so far, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) fn takes() -> u32 {
    TAKES.with(Cell::get)
}

/// The thread's outermost entry, whose retries the taken claims span,
/// has exited.
#[cold]
#[inline(never)]
pub(crate) fn clear_taken() {
    #[cfg(all(test, not(loom)))]
    TAKES.with(|n| n.set(0));
    crate::cycle_state::set_any_taken(false);
    TAKEN.with(|t| t.borrow_mut().clear());
    TAKEN_CYCLIC.with(|t| t.borrow_mut().clear());
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
