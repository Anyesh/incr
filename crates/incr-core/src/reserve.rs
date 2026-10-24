//! Doom reservations under `Shared`.
//!
//! A doom releases claims whose nodes the dooming thread's replay contexts
//! compute again, where the unbounded run keeps them claimed on that
//! thread's stack. A reservation keeps every other thread from computing
//! such a node until the thread's outermost context of that runtime, which
//! outlives every context nested in it, is removed; the dooming thread's
//! own claims, landings and releases of the node keep it, so a node a set
//! dirties again meanwhile waits for that context too. It is stored
//! before the release that publishes it, so a claim that reads the release
//! sees it. A node generation has at most one reservation: reserving it
//! needs its claim, and a claim waits out another thread's reservation.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use crate::fx::FxHasher;
use crate::waits::Claim;

#[cfg(not(loom))]
use std::sync::{
    atomic::{AtomicU32, AtomicUsize, Ordering},
    Mutex,
};

#[cfg(loom)]
use loom::sync::{
    atomic::{AtomicU32, AtomicUsize, Ordering},
    Mutex,
};

/// Slots of [`BUCKETS`]; a power of two.
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
static TABLE: Mutex<BTreeMap<Claim, Entry>> = Mutex::new(BTreeMap::new());

#[cfg(loom)]
loom::lazy_static! {
    static ref LIVE: AtomicUsize = AtomicUsize::new(0);
    static ref BUCKETS: Vec<AtomicU32> = (0..BUCKETS_LEN).map(|_| AtomicU32::new(0)).collect();
    static ref TABLE: Mutex<BTreeMap<Claim, Entry>> = Mutex::new(BTreeMap::new());
}

/// One context's reservations on this thread, as claim and generation.
struct Group {
    context: u64,
    claims: Vec<(Claim, u32)>,
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
            unlist(&g.claims, token);
        }
    }
}

#[cfg(not(loom))]
thread_local! {
    static MINE: Listed = const { Listed(RefCell::new(Vec::new())) };
    static SCOPE: Cell<Option<u64>> = const { Cell::new(None) };
}

#[cfg(loom)]
loom::thread_local! {
    static MINE: Listed = Listed(RefCell::new(Vec::new()));
    static SCOPE: Cell<Option<u64>> = Cell::new(None);
}

/// The context of a doom's reservations until the doom has marked one.
pub(crate) const PENDING: u64 = 0;

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

fn bucket(claim: Claim) -> &'static AtomicU32 {
    let mut h = FxHasher::default();
    claim.hash(&mut h);
    &BUCKETS[h.finish() as usize & (BUCKETS_LEN - 1)]
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
/// [`Scope`] is open and this thread does not reserve it already. Called
/// before the release.
pub(crate) fn note(claim: Claim, generation: u32) {
    let Some(context) = SCOPE.with(Cell::get) else {
        return;
    };
    MINE.with(|m| {
        let token = m.token();
        {
            let mut t = table();
            match t.insert(claim, (generation, token)) {
                Some(prev) if prev == (generation, token) => return,
                Some(_) => {}
                None => {
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
                claims: vec![(claim, generation)],
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

/// Whether another thread reserved `claim` at `generation`.
#[cold]
#[inline(never)]
pub(crate) fn elsewhere(claim: Claim, generation: u32) -> bool {
    if !any() || bucket(claim).load(Ordering::Relaxed) == 0 {
        return false;
    }
    let token = MINE.with(Listed::token);
    table()
        .get(&claim)
        .is_some_and(|&(g, owner)| g == generation && owner != token)
}

fn unlist(claims: &[(Claim, u32)], token: usize) {
    let mut t = table();
    for &(claim, generation) in claims {
        if t.get(&claim) == Some(&(generation, token)) {
            t.remove(&claim);
            bucket(claim).fetch_sub(1, Ordering::Relaxed);
        }
    }
    LIVE.store(t.len(), Ordering::Relaxed);
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
        CONTEXT_ENDS.with(|n| n.set(n.get() + g.claims.len() as u32));
        unlist(&g.claims, token);
    }
}

/// Ends every reservation of this thread, PENDING ones included: none of
/// its contexts outlives this.
pub(crate) fn end_all() {
    let (groups, token) = MINE.with(|m| (std::mem::take(&mut *m.0.borrow_mut()), m.token()));
    for g in groups {
        unlist(&g.claims, token);
    }
}

/// The claims this thread has reserved, which its claim waits list as
/// held.
pub(crate) fn held() -> Vec<Claim> {
    MINE.with(|m| {
        m.0.borrow()
            .iter()
            .flat_map(|g| g.claims.iter().map(|&(c, _)| c))
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
