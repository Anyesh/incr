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
use std::collections::{BTreeMap, HashMap, HashSet};
#[cfg(not(loom))]
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::attempt::{self, Attempt};
use crate::fx::FxBuildHasher;
#[cfg(not(loom))]
use crate::fx::FxHasher;
use crate::handle::RuntimeId;
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

type AwakeWalks = HashMap<(Claim, u32), u32, FxBuildHasher>;

/// Entries in [`TABLE`], and per hash bucket of their claims, both stored
/// under its lock: a claim that reads a release sees non-zero counts
/// without taking the lock, and takes it only on its own bucket's hit.
#[cfg(not(loom))]
static LIVE: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(loom))]
static BUCKETS: [AtomicU32; BUCKETS_LEN] = [const { AtomicU32::new(0) }; BUCKETS_LEN];

#[cfg(not(loom))]
static TABLE: Mutex<BTreeMap<Claim, (Entry, Attempt)>> = Mutex::new(BTreeMap::new());

/// Every thread's awake walk listings ([`Release`]) by claim and node
/// generation, counted once per release that lists one, so another thread
/// can tell a listing that holds every op ([`walk_listed_awake`]).
#[cfg(not(loom))]
static AWAKE_WALKS: Mutex<AwakeWalks> = Mutex::new(HashMap::with_hasher(FxBuildHasher));

/// Orders the start of each op that leaves the Clean fast path against
/// [`rest`]s.
#[cfg(not(loom))]
static SEQ: AtomicU64 = AtomicU64::new(1);

#[cfg(loom)]
loom::lazy_static! {
    static ref LIVE: AtomicUsize = AtomicUsize::new(0);
    static ref TABLE: Mutex<BTreeMap<Claim, (Entry, Attempt)>> = Mutex::new(BTreeMap::new());
    static ref AWAKE_WALKS: Mutex<AwakeWalks> = Mutex::new(HashMap::with_hasher(FxBuildHasher));
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
    /// [`SEQ`] as this thread's current op left the Clean fast path; 0
    /// before its first.
    static OP: Cell<u64> = const { Cell::new(0) };
    /// This thread's awake walk listings, by the release that made them.
    static WOKEN: RefCell<Vec<Release>> = const { RefCell::new(Vec::new()) };
    static ANY_WOKEN: Cell<bool> = const { Cell::new(false) };
}

#[cfg(loom)]
loom::thread_local! {
    static TAKEN: RefCell<Vec<Claim>> = RefCell::new(Vec::new());
    static TAKEN_CYCLIC: RefCell<Vec<(Claim, u32)>> = RefCell::new(Vec::new());
    static MINE: Listed = Listed(RefCell::new(Vec::new()));
    static SCOPE: Cell<Option<u64>> = Cell::new(None);
    static ROOT: Cell<u64> = Cell::new(0);
    static ROOTS: Cell<u32> = Cell::new(0);
    static OP: Cell<u64> = Cell::new(0);
    static WOKEN: RefCell<Vec<Release>> = RefCell::new(Vec::new());
    static ANY_WOKEN: Cell<bool> = Cell::new(false);
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

fn awake_walks() -> impl std::ops::DerefMut<Target = AwakeWalks> {
    #[cfg(not(loom))]
    {
        AWAKE_WALKS.lock().unwrap_or_else(|e| e.into_inner())
    }
    #[cfg(loom)]
    {
        AWAKE_WALKS.lock().unwrap()
    }
}

fn wake_walk(claim: (Claim, u32)) {
    *awake_walks().entry(claim).or_default() += 1;
}

fn end_awake_walks(claims: impl IntoIterator<Item = (Claim, u32)>) {
    let mut awake = awake_walks();
    for claim in claims {
        if let Some(n) = awake.get_mut(&claim) {
            *n -= 1;
            if *n == 0 {
                awake.remove(&claim);
            }
        }
    }
}

/// Whether some thread's walk listing of `claim` at `generation` is awake:
/// the unbounded run still holds the node in its owner's frame, so a
/// waiter may not wound the owner over it.
#[cold]
#[inline(never)]
pub(crate) fn walk_listed_awake(claim: Claim, generation: u32) -> bool {
    awake_walks().contains_key(&(claim, generation))
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
    if taken(claim) {
        return;
    }
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

/// The slots of one unwind's release, sorted: the nodes whose frames the
/// unbounded run holds in one episode. Shared by the release's listings.
#[derive(Clone)]
pub(crate) struct Peers(Rc<[u32]>);

impl Peers {
    pub(crate) fn new(slots: impl IntoIterator<Item = u32>) -> Self {
        let mut slots: Vec<u32> = slots.into_iter().collect();
        slots.sort_unstable();
        slots.dedup();
        Self(slots.into())
    }

    /// These and `slot`, the same set if it is one of them.
    pub(crate) fn and(self, slot: u32) -> Self {
        if self.contains(slot) {
            self
        } else {
            Self::new(self.0.iter().copied().chain([slot]))
        }
    }

    fn contains(&self, slot: u32) -> bool {
        self.0.binary_search(&slot).is_ok()
    }
}

/// One release's walk listings still awake: claims the walk group of
/// `context` lists, each with the stamp this thread's other reservation
/// of the node had before the first listing of it woke it, or since
/// ([`AWAKE`] for none or awake), the same in every release that lists it,
/// kept awake through the closures of `peers` of `rt`.
struct Release {
    context: u64,
    rt: RuntimeId,
    peers: Peers,
    woken: HashMap<(Claim, u32), u64, FxBuildHasher>,
}

/// Reserves `claim`, which a user panic's unwind inside this thread's
/// innermost read just landed New or Dirty, until the read lands it
/// settled or ends. A walk entry's contained unwind lands claims its walk
/// runs again outside any doom's scope, and the context whose end follows
/// the containment no longer holds them. The listing holds every op while
/// the unbounded run would still hold the node in its frame: through
/// engine code and the closures of `peers`, the slots the unwind released
/// with it, whose frames sit in the same episode. It ends where this
/// thread next enters another closure ([`user_code`]) or any cycle
/// callback ([`callback_code`]), where the unbounded run holds no frame
/// of it, so a reader that code joins, begun before or after, passes. A
/// reservation of the node this thread already held goes back to its
/// stamp. The read's own node is [`note_root`]'s.
#[cold]
#[inline(never)]
pub(crate) fn note_walk(claim: Claim, generation: u32, peers: &Peers) {
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
        let mut was = AWAKE;
        {
            let mut t = table();
            match t.get_mut(&claim) {
                // Another thread's reservation stands for the node; at
                // rest it holds fewer ops than this listing would, the known
                // later-op gap.
                Some(((g, owner, _), _)) if *g == generation && *owner != token => return None,
                Some(((g, _, stamp), a)) if *g == generation => {
                    *a = attempt::mine();
                    was = std::mem::replace(stamp, AWAKE);
                }
                _ => {
                    if t.insert(claim, ((generation, token, AWAKE), attempt::mine()))
                        .is_none()
                    {
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
        Some(was)
    });
    let Some(was) = listed else {
        return;
    };
    ROOTS.with(|n| n.set(n.get() + 1));
    WOKEN.with(|w| {
        let mut w = w.borrow_mut();
        let same = |r: &Release| r.context == context && Rc::ptr_eq(&r.peers.0, &peers.0);
        if !w.last().is_some_and(same) {
            w.push(Release {
                context,
                rt: claim.0,
                peers: peers.clone(),
                woken: HashMap::default(),
            });
        }
        // A listing by an enclosing read already woke it: the stamp under
        // both is the one that read saw.
        let was = w
            .iter()
            .find_map(|r| r.woken.get(&(claim, generation)).copied())
            .unwrap_or(was);
        let release = w.last_mut().expect("a release was just pushed");
        if release.woken.insert((claim, generation), was).is_none() {
            wake_walk((claim, generation));
        }
    });
    ANY_WOKEN.with(|a| a.set(true));
    #[cfg(all(test, not(loom)))]
    WALK_LISTINGS.with(|n| n.set(n.get() + 1));
}

/// Whether this thread is inside a read, whose walk [`note_walk`] lists
/// what an unwind lands unsettled.
pub(crate) fn walking() -> bool {
    ROOT.with(Cell::get) != 0
}

/// This thread is about to run the closure of `entering`: its walk
/// listings end here unless `entering` is one of their peers.
#[inline(always)]
pub(crate) fn user_code(entering: Claim) {
    if ANY_WOKEN.with(Cell::get) {
        end_woken(Some(entering));
    }
}

/// This thread is about to run a cycle callback: its walk listings end
/// here, peers' callbacks too, since the callback may join a thread that
/// reads a listed node while the unbounded run holds no frame of it.
#[inline(always)]
pub(crate) fn callback_code() {
    if ANY_WOKEN.with(Cell::get) {
        end_woken(None);
    }
}

/// Ends the walk listings of every release `entering` is no peer of.
#[cold]
#[inline(never)]
fn end_woken(entering: Option<Claim>) {
    let mut ended = Vec::new();
    WOKEN.with(|w| {
        let mut w = w.borrow_mut();
        #[cfg(all(test, not(loom)))]
        WALK_WORK.with(|n| n.set(n.get() + w.len() as u64));
        w.retain_mut(|r| {
            let keep = entering.is_some_and(|(rt, slot)| r.rt == rt && r.peers.contains(slot));
            if !keep {
                ended.extend(r.woken.drain().map(|(c, was)| (r.context, c, was)));
            }
            keep
        });
        ANY_WOKEN.with(|a| a.set(!w.is_empty()));
    });
    #[cfg(all(test, not(loom)))]
    WALK_WORK.with(|n| n.set(n.get() + ended.len() as u64));
    end_awake_walks(ended.iter().map(|&(_, claim, _)| claim));
    for (context, claim, was) in ended {
        end_walk(context, claim, was);
    }
}

/// Ends this thread's walk listings of `claim`, which its run just landed
/// settled, as the unbounded run's frame releases its claim there.
#[cold]
#[inline(never)]
pub(crate) fn end_walk_landed(claim: Claim, generation: u32) {
    if ROOTS.with(Cell::get) == 0 {
        return;
    }
    let claim = (claim, generation);
    let mut ended = Vec::new();
    WOKEN.with(|w| {
        let mut w = w.borrow_mut();
        w.retain_mut(|r| {
            if let Some(was) = r.woken.remove(&claim) {
                ended.push((r.context, was));
            }
            !r.woken.is_empty()
        });
        ANY_WOKEN.with(|a| a.set(!w.is_empty()));
    });
    end_awake_walks(ended.iter().map(|_| claim));
    for (context, was) in ended {
        end_walk(context, claim, was);
    }
}

/// Takes `claim` out of this thread's walk group of `context`, whose
/// listing of it ended, leaving a reservation another group of the thread
/// holds at `was` once no walk group lists it.
fn end_walk(context: u64, claim: (Claim, u32), was: u64) {
    let ended = MINE.try_with(|m| {
        let mut groups = m.0.borrow_mut();
        let at = groups.iter().position(|g| g.context == context)?;
        if !groups[at].claims.remove(&claim) {
            return None;
        }
        if groups[at].claims.is_empty() {
            groups.swap_remove(at);
        }
        Some(m.token())
    });
    if let Ok(Some(token)) = ended {
        ROOTS.with(|n| n.set(n.get() - 1));
        unlist([claim], token, true);
        if !walk_listed(claim) {
            restamp([(claim, was)], token);
        }
    }
}

/// Puts back the stamp each of this thread's reservations had before a
/// walk listing woke it, where another group still holds it awake.
fn restamp(woken: impl IntoIterator<Item = ((Claim, u32), u64)>, token: usize) {
    let mut t = table();
    for ((claim, generation), was) in woken {
        if was == AWAKE {
            continue;
        }
        if let Some((e, _)) = t.get_mut(&claim) {
            if (e.0, e.1) == (generation, token) && e.2 == AWAKE {
                e.2 = was;
            }
        }
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
        let mut woken = Vec::new();
        let _ = WOKEN.try_with(|w| {
            let mut w = w.borrow_mut();
            w.retain_mut(|r| {
                let keep = r.context != (WALK | key);
                if !keep {
                    woken.extend(r.woken.drain());
                }
                keep
            });
            let _ = ANY_WOKEN.try_with(|a| a.set(!w.is_empty()));
        });
        end_awake_walks(woken.iter().map(|&(claim, _)| claim));
        unlist(ended, token, true);
        woken.retain(|&(claim, _)| !walk_listed(claim));
        restamp(woken, token);
    }
}

/// Whether a walk group of this thread lists `claim`.
fn walk_listed(claim: (Claim, u32)) -> bool {
    MINE.try_with(|m| {
        m.0.borrow()
            .iter()
            .any(|g| g.context & WALK != 0 && g.context != ROOT_READ && g.claims.contains(&claim))
    })
    .unwrap_or(false)
}

/// Sets the stamp under each awake walk listing of `claim` to `stamp`,
/// where `only_awake` keeps one already set. Whether any listing does.
fn set_woken_stamp(claim: (Claim, u32), stamp: impl Fn() -> u64, only_awake: bool) -> bool {
    if !ANY_WOKEN.try_with(Cell::get).unwrap_or(false) {
        return false;
    }
    WOKEN
        .try_with(|w| {
            let mut found = false;
            for r in w.borrow_mut().iter_mut() {
                if let Some(was) = r.woken.get_mut(&claim) {
                    if !only_awake || *was == AWAKE {
                        *was = stamp();
                    }
                    found = true;
                }
            }
            found
        })
        .unwrap_or(false)
}

fn list(context: u64, claim: Claim, generation: u32, again: bool) {
    // The reservation under an awake listing is awake again, whatever an
    // earlier context's rest left there.
    set_woken_stamp((claim, generation), || AWAKE, false);
    MINE.with(|m| {
        let token = m.token();
        {
            let mut t = table();
            match t.insert(claim, ((generation, token, AWAKE), attempt::mine())) {
                Some(((g, owner, _), _)) if (g, owner) == (generation, token) && !again => return,
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
        .filter(|((g, _, _), a)| *g == generation && !a.wounded())
        .map(|&((_, owner, stamp), _)| (owner == token, stamp))
}

/// The attempt of another thread that reserved `claim` at `generation`.
fn owner_elsewhere(claim: Claim, generation: u32) -> Option<Attempt> {
    if !any() || !bucket_hit(claim) {
        return None;
    }
    let token = MINE.with(Listed::token);
    table()
        .get(&claim)
        .filter(|&&((g, owner, _), _)| g == generation && owner != token)
        .map(|(_, a)| a.clone())
}

/// Whether another thread reserved `claim` at `generation` against this
/// thread's op, for an attempt no waiter has wounded: a wounded attempt's
/// reservations hold no one. A reservation at rest holds only ops begun
/// before its owner left the node's run: a later op is a reader the
/// owner's retry may itself have started, before that retry reads the
/// node again.
#[cold]
#[inline(never)]
pub(crate) fn elsewhere(claim: Claim, generation: u32) -> bool {
    matches!(reserved(claim, generation), Some((false, stamp)) if OP.with(Cell::get) <= stamp)
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
    let mut t = table();
    let Some((e, _)) = t.get_mut(&claim) else {
        return;
    };
    if (e.0, e.1) != (generation, token) || e.2 != AWAKE {
        return;
    }
    // An awake walk listing keeps the node awake; the rest takes effect
    // where the last one ends.
    let stamp = std::cell::OnceCell::new();
    let at = || *stamp.get_or_init(|| SEQ.fetch_add(1, Ordering::SeqCst));
    if !set_woken_stamp((claim, generation), at, true) {
        e.2 = at();
    }
}

/// Ends `claims`, already out of their group. `shared` says another
/// group may list them too, which only a read's reservation of its own
/// node makes possible.
fn unlist(claims: impl IntoIterator<Item = (Claim, u32)>, token: usize, shared: bool) {
    let mut t = table();
    for (claim, generation) in claims {
        let Some((e, _)) = t.get_mut(&claim) else {
            continue;
        };
        if (e.0, e.1) != (generation, token) {
            continue;
        }
        if !(shared && still_listed(claim, generation)) {
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

/// Whether another group of this thread still lists `claim`, as a read
/// of its own node or its walk and a context can both. False once the
/// thread's reservations are being destroyed.
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
    static WALK_LISTINGS: Cell<u32> = const { Cell::new(0) };
    static WALK_WORK: Cell<u64> = const { Cell::new(0) };
}

/// Releases looked at and listings ended at this thread's entries into
/// user code since the last call, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) fn take_walk_work() -> u64 {
    WALK_WORK.with(|n| n.replace(0))
}

/// Walk listings this thread made, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) fn walk_listings() -> u32 {
    WALK_LISTINGS.with(Cell::get)
}

/// Reservations this thread's context removals ended, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) fn context_ends() -> u32 {
    CONTEXT_ENDS.with(Cell::get)
}

/// Whether this thread holds `claim` awake, for tests.
#[cfg(all(test, not(loom)))]
pub(crate) fn mine_awake(claim: Claim) -> bool {
    let token = MINE.with(Listed::token);
    any()
        && matches!(table().get(&claim), Some(&((_, owner, stamp), _)) if owner == token && stamp == AWAKE)
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
    let woken = WOKEN.with(|w| std::mem::take(&mut *w.borrow_mut()));
    end_awake_walks(woken.into_iter().flat_map(|r| r.woken.into_keys()));
    ANY_WOKEN.with(|a| a.set(false));
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
