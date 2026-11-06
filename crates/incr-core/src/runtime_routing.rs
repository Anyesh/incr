//! Eager group routing. A `group_by` router is marked eager
//! ([`EAGER`] with `WATCHED`), so a dirty walk or a landing that reaches it
//! puts it on this thread's pending list for its runtime, and the call
//! that queued it routes the list before returning. A call routes only
//! when no entry of the runtime is open on the thread (an entry would
//! change how inner calls catch and unwind), no routing hold of the
//! runtime is live on the thread, no routing of it is already running on
//! the thread, and the thread is not panicking. A list left unrouted is
//! routed by the next call that passes those checks.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::marker::PhantomData;

use super::*;
use crate::segmented_nodes::{is_eager, EAGER};

type Router = (u32, u32);

/// What this thread owes one runtime: the routers its calls queued, in a
/// list deduplicated by `queued`, and the state of its holds and drains.
struct Owed {
    rt: RuntimeId,
    pending: Vec<Router>,
    queued: HashSet<Router, FxBuildHasher>,
    holds: u32,
    draining: bool,
    failure: Option<RoutingFailed>,
}

impl Owed {
    fn idle(&self) -> bool {
        self.pending.is_empty() && self.holds == 0 && !self.draining && self.failure.is_none()
    }
}

thread_local! {
    static OWED: RefCell<Vec<Owed>> = const { RefCell::new(Vec::new()) };
    /// The routers queued on this thread across runtimes, so that a call
    /// tells with one load that it owes nothing.
    static QUEUED: Cell<usize> = const { Cell::new(0) };
}

/// Runs `f` on this thread's entry for `rt`, made if `make`. Nothing runs
/// once the thread's locals are gone.
fn with_owed<R>(rt: RuntimeId, make: bool, f: impl FnOnce(&mut Owed) -> R) -> Option<R> {
    OWED.try_with(|owed| {
        let mut owed = owed.borrow_mut();
        let at = match owed.iter().position(|o| o.rt == rt) {
            Some(at) => at,
            None if make => {
                owed.push(Owed {
                    rt,
                    pending: Vec::new(),
                    queued: HashSet::default(),
                    holds: 0,
                    draining: false,
                    failure: None,
                });
                owed.len() - 1
            }
            None => return None,
        };
        let r = f(&mut owed[at]);
        if owed[at].idle() {
            owed.swap_remove(at);
        }
        Some(r)
    })
    .ok()
    .flatten()
}

#[inline(always)]
fn any_queued() -> bool {
    QUEUED.with(Cell::get) != 0
}

fn queue<C: Cells>(rt: &Runtime<C>, router: Router) {
    with_owed(rt.runtime_id, true, |owed| {
        if owed.queued.insert(router) {
            owed.pending.push(router);
            QUEUED.with(|q| q.set(q.get() + 1));
            rt.routing_owing.add(1);
        }
    });
}

/// Drops `router` from the list, where its own landing put it back.
fn unqueue<C: Cells>(rt: &Runtime<C>, router: Router) {
    with_owed(rt.runtime_id, false, |owed| {
        if owed.queued.remove(&router) {
            owed.pending.retain(|&r| r != router);
            QUEUED.with(|q| q.set(q.get() - 1));
            rt.routing_owing.sub(1);
        }
    });
}

fn pop<C: Cells>(rt: &Runtime<C>) -> Option<Router> {
    with_owed(rt.runtime_id, false, |owed| {
        let router = owed.pending.pop()?;
        owed.queued.remove(&router);
        QUEUED.with(|q| q.set(q.get() - 1));
        rt.routing_owing.sub(1);
        Some(router)
    })
    .flatten()
}

/// The thread's routers queued for `rt`.
#[cfg(all(test, not(loom)))]
pub(crate) fn queued_routers<C: Cells>(rt: &Runtime<C>) -> usize {
    with_owed(rt.runtime_id, false, |owed| owed.pending.len()).unwrap_or(0)
}

/// The routers queued and failures stored for `rt`, on every thread.
#[cfg(all(test, not(loom)))]
pub(crate) fn owed_to<C: Cells>(rt: &Runtime<C>) -> u64 {
    C::u64_load_relaxed(&rt.routing_owing.owed)
}

/// What a runtime's calls owe it on every thread: routers queued and
/// failures stored, so that a call finding none returns with one load of
/// the runtime it already holds (a thread-local in a shared library is a
/// call). Also the routing holds of a `Local` runtime, which only one
/// thread uses; a `Shared` runtime keeps its holds per thread in `Owed`.
pub(crate) struct Owing<C: Cells> {
    owed: C::U64,
    local_holds: C::U32,
}

impl<C: Cells> Owing<C> {
    pub(crate) fn new() -> Self {
        Self {
            owed: C::new_u64(0),
            local_holds: C::new_u32(0),
        }
    }

    #[inline(always)]
    fn none(&self) -> bool {
        C::u64_load_relaxed(&self.owed) == 0
    }

    fn add(&self, n: u64) {
        C::u64_fetch_add(&self.owed, n);
    }

    fn sub(&self, n: u64) {
        let was = C::u64_fetch_add(&self.owed, n.wrapping_neg());
        debug_assert!(was >= n, "a runtime's owed count went below zero");
    }
}

/// Marks a drain of `rt` running on this thread for its lifetime, an
/// unwind's included.
struct Draining(RuntimeId);

impl Draining {
    /// A drain of `rt`, unless a hold or a drain of it is live on this
    /// thread or nothing is queued for it.
    fn begin<C: Cells>(rt: &Runtime<C>) -> Option<Self> {
        if C::u32_load_relaxed(&rt.routing_owing.local_holds) != 0 {
            return None;
        }
        let rt = rt.runtime_id;
        with_owed(rt, false, |owed| {
            if owed.holds != 0 || owed.draining || owed.pending.is_empty() {
                return None;
            }
            owed.draining = true;
            Some(Draining(rt))
        })
        .flatten()
    }
}

impl Drop for Draining {
    fn drop(&mut self) {
        with_owed(self.0, false, |owed| owed.draining = false);
    }
}

/// A router popped for a drain, queued again if an unwind leaves it
/// unrouted.
struct Popped<'a, C: Cells>(&'a Runtime<C>, Router);

impl<C: Cells> Drop for Popped<'_, C> {
    fn drop(&mut self) {
        queue(self.0, self.1);
    }
}

/// A routing run that failed: the router, left Failed, and the payload of
/// the panic that failed it. A read of the router raises it as today; the
/// groups keep their last values until a later routing succeeds.
pub struct RoutingFailed {
    router: ErasedIncr,
    payload: Box<dyn Any + Send>,
}

impl RoutingFailed {
    pub fn router(&self) -> ErasedIncr {
        self.router
    }

    /// The panic message, when the payload is a string.
    pub fn message(&self) -> Option<&str> {
        self.payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| self.payload.downcast_ref::<&str>().copied())
    }

    pub fn payload(&self) -> &(dyn Any + Send) {
        &*self.payload
    }

    pub fn into_payload(self) -> Box<dyn Any + Send> {
        self.payload
    }
}

impl fmt::Debug for RoutingFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RoutingFailed")
            .field("router", &self.router)
            .field("message", &self.message())
            .finish()
    }
}

impl fmt::Display for RoutingFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "incr-core: routing the group_by router at slot {} failed: {}",
            self.router.slot(),
            self.message()
                .unwrap_or("a panic with a non-string payload")
        )
    }
}

impl std::error::Error for RoutingFailed {}

/// A routing hold: while one is live on a thread, no call on that thread
/// routes its runtime's routers. Holds nest; when the outermost one drops,
/// a hold made by [`Runtime::hold_routing`] routes what was queued
/// meanwhile, and one made by [`Runtime::hold_routing_suppressed`] leaves
/// it for the next call that routes, or for [`Runtime::route_pending`].
///
/// A layer that holds its own lock across writes takes a hold before the
/// lock and drops it after, since routing runs user code (`key_fn` and
/// the router's upstream chain) that may need the lock.
#[must_use]
pub struct RoutingHold<'a, C: Cells> {
    rt: &'a Runtime<C>,
    routes: bool,
    _thread: PhantomData<*const ()>,
}

impl<C: Cells> Drop for RoutingHold<'_, C> {
    fn drop(&mut self) {
        let last = if C::SINGLE_THREADED {
            let holds = &self.rt.routing_owing.local_holds;
            let left = C::u32_load_relaxed(holds) - 1;
            C::u32_store_relaxed(holds, left);
            Some(left == 0)
        } else {
            with_owed(self.rt.runtime_id, false, |owed| {
                owed.holds -= 1;
                owed.holds == 0
            })
        };
        if last == Some(true) && self.routes {
            self.rt.route_queued();
        }
    }
}

/// A runtime's record of which routers each router's runs dirtied, so a
/// thread that waited on another thread's routing routes, or waits on, the
/// nested routing too. Lists are cumulative: a run that sets nothing must
/// not erase what an earlier run owes. An entry goes when a drain finds
/// its router Clean, or when its owner is deleted.
pub(crate) struct RouterLists<C: Cells> {
    rt: RuntimeId,
    lists: OrderedLock<HashMap<Router, HashSet<Router, FxBuildHasher>, FxBuildHasher>, C>,
}

impl<C: Cells> RouterLists<C> {
    pub(crate) fn new(rt: RuntimeId) -> Self {
        Self {
            rt,
            lists: OrderedLock::new(LockKind::RouterLists, HashMap::default()),
        }
    }
}

impl<C: Cells> Drop for RouterLists<C> {
    fn drop(&mut self) {
        let forgotten = OWED.try_with(|owed| {
            let mut owed = owed.borrow_mut();
            let at = owed.iter().position(|o| o.rt == self.rt)?;
            let gone = owed.swap_remove(at);
            QUEUED.with(|q| q.set(q.get() - gone.pending.len()));
            Some(gone)
        });
        drop(forgotten);
    }
}

impl<C: Cells> Runtime<C> {
    /// A routing hold that routes when the outermost hold on this thread
    /// drops ([`RoutingHold`]).
    pub fn hold_routing(&self) -> RoutingHold<'_, C> {
        self.routing_hold(true)
    }

    /// A routing hold that never routes when it drops: writes under it
    /// only queue routers, for [`Self::route_pending`] or the next call
    /// that routes.
    pub fn hold_routing_suppressed(&self) -> RoutingHold<'_, C> {
        self.routing_hold(false)
    }

    fn routing_hold(&self, routes: bool) -> RoutingHold<'_, C> {
        if C::SINGLE_THREADED {
            let holds = &self.routing_owing.local_holds;
            C::u32_store_relaxed(holds, C::u32_load_relaxed(holds) + 1);
        } else {
            with_owed(self.runtime_id, true, |owed| owed.holds += 1);
        }
        RoutingHold {
            rt: self,
            routes,
            _thread: PhantomData,
        }
    }

    /// Routes the routers this thread's calls queued for this runtime,
    /// unless an entry of it is open on the thread, a routing hold or a
    /// drain of it is live on the thread, the thread is in value code, or
    /// it is panicking. Returns the first failure of a routing run this
    /// call started; later ones are dropped.
    #[inline]
    pub fn route_pending(&self) -> Option<RoutingFailed> {
        if self.routing_owing.none() {
            return None;
        }
        self.route_pending_owed()
    }

    #[cold]
    #[inline(never)]
    fn route_pending_owed(&self) -> Option<RoutingFailed> {
        if cycle_state::open_entry_of(self.runtime_id) {
            return None;
        }
        self.drain_gated()
    }

    /// The first routing failure a call on this thread stored since the
    /// last take, cleared by the take. `None` while an entry of this
    /// runtime is open on the thread, so a read inside a compute leaves it
    /// for the next top-level call.
    #[inline]
    pub fn take_routing_failure(&self) -> Option<RoutingFailed> {
        if self.routing_owing.none() {
            return None;
        }
        self.take_stored_failure()
    }

    #[cold]
    #[inline(never)]
    fn take_stored_failure(&self) -> Option<RoutingFailed> {
        if cycle_state::open_entry_of(self.runtime_id) {
            return None;
        }
        let taken = with_owed(self.runtime_id, false, |owed| owed.failure.take()).flatten();
        if taken.is_some() {
            self.routing_owing.sub(1);
        }
        taken
    }

    /// At the exit of a call that may have queued a router: route, keeping
    /// a failure for [`Self::take_routing_failure`].
    #[inline(always)]
    pub(crate) fn route_if_queued(&self) {
        if any_queued() {
            self.route_queued();
        }
    }

    #[cold]
    #[inline(never)]
    pub(crate) fn route_queued(&self) {
        if !cycle_state::open_entry_of(self.runtime_id) {
            self.route_storing();
        }
    }

    /// [`Self::route_if_queued`] inside `stabilize`'s own entry, which
    /// found no entry open before it made its own.
    #[inline(always)]
    pub(super) fn route_in_stabilize(&self) {
        if any_queued() {
            self.route_storing();
        }
    }

    #[cold]
    #[inline(never)]
    fn route_storing(&self) {
        if let Some(failure) = self.drain_gated() {
            let lost = with_owed(self.runtime_id, true, |owed| match owed.failure {
                None => {
                    owed.failure = Some(failure);
                    self.routing_owing.add(1);
                    None
                }
                Some(_) => Some(failure),
            });
            drop(lost);
        }
    }

    fn drain_gated(&self) -> Option<RoutingFailed> {
        if std::thread::panicking() || value_code::thread_word() & value_code::COUNTER_MASK != 0 {
            return None;
        }
        let _draining = Draining::begin(self)?;
        crate::lock_order::assert_may_route();
        let mut first = None;
        while let Some(router) = pop(self) {
            let popped = Popped(self, router);
            let mut visited = HashSet::default();
            self.route_one(router, &mut first, &mut visited);
            std::mem::forget(popped);
        }
        first
    }

    /// Routes `router`, then on `Shared` the routers its runs dirtied, which
    /// another thread's run of it may have queued on that thread instead.
    fn route_one(
        &self,
        (slot, generation): Router,
        first: &mut Option<RoutingFailed>,
        visited: &mut HashSet<Router, FxBuildHasher>,
    ) {
        let handle = Incr::<u64>::new(slot, generation, self.runtime_id);
        if let Err(payload) = self.route_walk(handle) {
            if first.is_none() {
                *first = Some(RoutingFailed {
                    router: handle.erase(),
                    payload,
                });
            }
        }
        // A run a cycle refused lands Dirty or New, which queues the router
        // again, and routing it again would meet the same cycle forever. A
        // write that flagged it during the run queued it on its own thread.
        let node = self.nodes.get(slot);
        if node.generation() == generation && node.state() != NodeState::Clean {
            unqueue(self, (slot, generation));
        }
        if C::SINGLE_THREADED {
            return;
        }
        let dirtied: Vec<Router> = self
            .router_lists
            .lists
            .read()
            .get(&(slot, generation))
            .map_or_else(Vec::new, |list| list.iter().copied().collect());
        for nested in dirtied {
            if visited.insert(nested) {
                self.route_one(nested, first, visited);
            }
            self.prune_routed((slot, generation), nested);
        }
    }

    /// Drops `nested` from `router`'s list if it is Clean or gone. Checked
    /// under the lists' lock, which a walk takes after its Dirty store, so
    /// an entry a newer run recorded is never pruned.
    fn prune_routed(&self, router: Router, nested: Router) {
        let mut lists = self.router_lists.lists.write();
        let node = self.nodes.get(nested.0);
        if node.generation() == nested.1 && node.state() != NodeState::Clean {
            return;
        }
        if let Some(list) = lists.get_mut(&router) {
            list.remove(&nested);
            if list.is_empty() {
                lists.remove(&router);
            }
        }
    }

    /// Creates a `group_by` router: eager, and watched so the walk reaches
    /// it through its watched arms.
    pub(crate) fn create_router<F>(&self, f: F) -> Incr<u64>
    where
        F: Fn(&Runtime<C>) -> u64 + Send + Sync + 'static,
    {
        self.create_query_erased::<true, _, _>(
            f,
            ErasedOpts {
                strategy: None,
                flags: EAGER | WATCHED,
            },
        )
    }

    /// A new router's first run, which records its upstream edge. A failure
    /// leaves it Failed with that edge, so the next write reaching it routes
    /// again.
    pub(crate) fn route_created(&self, router: Incr<u64>) {
        drop(self.route_walk(router));
    }

    /// Queues `id`, an eager node a walk or landing reached. A walk inside
    /// a router's run also notes it on that router's list (`Shared`).
    #[cold]
    #[inline(never)]
    pub(super) fn queue_eager(&self, id: NodeId, node: &NodeData<C>, in_walk: bool) {
        let router = (id.0, node.generation());
        queue(self, router);
        if !C::SINGLE_THREADED && in_walk {
            self.note_in_running_router(router);
        }
    }

    fn note_in_running_router(&self, dirtied: Router) {
        let Some((top, running, _)) = self.dep_stack.top_for_lookup() else {
            return;
        };
        if top.rt() != self.runtime_id || running.0 == dirtied.0 {
            return;
        }
        let (node, flags) = self.nodes.get_with_flags(running.0);
        if is_eager(C::u8_load_relaxed(flags)) {
            let owner = (running.0, node.generation());
            self.router_lists
                .lists
                .write()
                .entry(owner)
                .or_default()
                .insert(dirtied);
        }
    }

    /// The delete's part for a router, under its write hold: its list goes.
    #[cold]
    #[inline(never)]
    pub(super) fn forget_router(&self, slot: u32, generation: u32) {
        if !C::SINGLE_THREADED {
            self.router_lists.lists.write().remove(&(slot, generation));
        }
    }
}
