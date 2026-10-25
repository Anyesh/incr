//! Owned children: `child_input` and `child_query`, the index a
//! creator's runs reach through their control block, and what each
//! landing of a creator does with the children its claim created or
//! left untouched.

use std::collections::HashSet;
use std::hash::Hash;

use super::*;

/// What a creator's landing does with its children.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ChildLanding {
    /// A Clean landing after a run: every child the last run did not
    /// touch, and every orphan, is deleted unless it is busy or has a
    /// dependent outside its owned set.
    Sweep,
    /// A landing that keeps the attempt's value keeps its children.
    Keep,
    /// A landing that discards the attempt: the children its claim
    /// created are unbound from their keys, so no later run reuses a
    /// value a discarded attempt chose, and deleted where they can be.
    Discard,
    /// A speculative run's landing whose stash holds the run: its children
    /// stay the claim in progress's, for the stash's adoption or drop.
    Stashed,
}

impl<C: Cells> Runtime<C> {
    /// The input child named `key` of the compute running on this
    /// runtime, created with `init()` and owned by that compute's node
    /// the first time it asks. Later runs that ask for `key` get the same
    /// node, and `init` is dropped unrun: a child is never updated by its
    /// creator, so a value chosen by an attempt that is later discarded
    /// never reaches a live input. A child's identity is its creator,
    /// `key`, `K`, `T` and its kind (input or query).
    ///
    /// A Clean landing of the creator deletes the children its last run
    /// did not ask for, unless one is busy or read by a node outside its
    /// own owned set; such a child stays and is checked again at the next
    /// Clean landing. A landing that discards the run deletes the
    /// children it created. Deleting the creator deletes its children.
    ///
    /// Panics outside a compute of this runtime.
    pub fn child_input<K, T>(&self, key: K, init: impl FnOnce() -> T) -> Incr<T>
    where
        K: Hash + Eq + Send + Sync + 'static,
        T: Value,
    {
        let (creator, ctl) = self.creator("child_input");
        if let Some(hit) = self.child_hit::<K, T, false>(creator, ctl, &key) {
            return hit;
        }
        if self.frame_doomed() {
            return self.pending_handle();
        }
        let child = self.create_input_unchecked(init());
        self.attach_child::<K, T, false>(creator, ctl, key, child.erase());
        child
    }

    /// The query child named `key` of the running compute, created with
    /// `f` the first time, like [`Self::child_input`]. On a hit `f` is
    /// dropped and the child keeps the closure it was created with.
    ///
    /// Panics outside a compute of this runtime.
    pub fn child_query<K, T, F>(&self, key: K, f: F) -> Incr<T>
    where
        K: Hash + Eq + Send + Sync + 'static,
        T: Value,
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.child_query_erased(key, f, || ErasedOpts::NONE, "child_query")
    }

    /// [`Self::child_query`] with options: the child's cycle strategy,
    /// used only when the call creates the child.
    pub fn child_query_with<K, T, F>(&self, key: K, f: F, opts: QueryOpts<T>) -> Incr<T>
    where
        K: Hash + Eq + Send + Sync + 'static,
        T: Value,
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        self.child_query_erased(key, f, || self.erase_query_opts(opts), "child_query_with")
    }

    /// `opts` is erased only when the call creates the child.
    fn child_query_erased<K, T, F>(
        &self,
        key: K,
        f: F,
        opts: impl FnOnce() -> ErasedOpts,
        what: &str,
    ) -> Incr<T>
    where
        K: Hash + Eq + Send + Sync + 'static,
        T: Value,
        F: Fn(&Runtime<C>) -> T + Send + Sync + 'static,
    {
        let (creator, ctl) = self.creator(what);
        if let Some(hit) = self.child_hit::<K, T, true>(creator, ctl, &key) {
            return hit;
        }
        if self.frame_doomed() {
            return self.pending_handle();
        }
        let child = self.create_query_erased::<false, _, _>(f, opts());
        self.attach_child::<K, T, true>(creator, ctl, key, child.erase());
        child
    }

    /// The node and control block of this runtime's running compute.
    fn creator(&self, what: &str) -> (NodeId, *const ComputeCtl) {
        if value_code::checked_word() & value_code::REGEN_BIT != 0 {
            super::replay_impl::abort_regen();
        }
        match self.dep_stack.active_compute(self.runtime_id) {
            Some(frame) => frame,
            None => panic!(
                "incr-core: {what} called outside a compute; only a compute creates children"
            ),
        }
    }

    /// The live child bound to `key`, touched by this run. A binding to a
    /// child deleted since (by a delete of the child itself) is dropped,
    /// so the caller creates a fresh one.
    fn child_hit<K, T, const QUERY: bool>(
        &self,
        creator: NodeId,
        ctl: *const ComputeCtl,
        key: &K,
    ) -> Option<Incr<T>>
    where
        K: Hash + Eq + Send + Sync + 'static,
        T: Value,
    {
        // SAFETY: the control block of the active frame is live while
        // the frame is on the stack, which outlasts this call.
        let index = self.open_index(creator, unsafe { &*ctl });
        if index.is_null() {
            return None;
        }
        // SAFETY: see `Self::index_at`.
        let index = unsafe { &*index };
        let found = value_code::run(|| index.with(|s| s.find::<K, T, QUERY>(key)))?;
        if self.is_alive(found) {
            return Some(Incr::new(found.slot(), found.generation(), self.runtime_id));
        }
        let unbound = value_code::run(|| index.with(|s| s.unbind::<K, T, QUERY>(key)));
        drop(unbound);
        None
    }

    /// Owns `child` by `creator` and binds it to `key`, starting the
    /// creator's index if this is its first child. The index is created
    /// and flagged under one write hold while this thread holds the
    /// creator's claim, so the next claimant, whose claim reads from this
    /// one's landing, finds the flag set when it reloads the flags.
    fn attach_child<K, T, const QUERY: bool>(
        &self,
        creator: NodeId,
        ctl: *const ComputeCtl,
        key: K,
        child: ErasedIncr,
    ) where
        K: Hash + Eq + Send + Sync + 'static,
        T: Value,
    {
        // SAFETY: as in `Self::child_hit`.
        let ctl = unsafe { &*ctl };
        let mut index = ctl.children.get().cast::<ChildIndex<C>>();
        {
            let mut inner = self.inner.write();
            if index.is_null() {
                let boxed = inner
                    .children
                    .entry(creator.0)
                    .or_insert_with(|| Box::new(ChildIndex::new()));
                index = &**boxed;
                let flags = self.nodes.get_with_flags(creator.0).1;
                C::u8_store_relaxed(flags, C::u8_load_relaxed(flags) | HAS_INDEX);
                ctl.children.set(index.cast());
            }
            self.adopt_locked(&mut inner, creator.0, child.slot());
        }
        // SAFETY: see `Self::index_at`.
        let index = unsafe { &*index };
        value_code::run(|| index.with(|s| s.insert::<K, T, QUERY>(key, child)));
    }

    /// The index this run of `creator` reaches, opened by its first
    /// child call: the run's number is bumped then, so the runs that call
    /// no child pay nothing. Null for a node that has created no child.
    /// The flags are loaded under the claim, which reads from the previous
    /// holder's landing, so a first child that another thread's run
    /// created is seen and no second index is started.
    #[inline]
    fn open_index(&self, creator: NodeId, ctl: &ComputeCtl) -> *const ChildIndex<C> {
        let open = ctl.children.get().cast::<ChildIndex<C>>();
        if !open.is_null() {
            return open;
        }
        let flags = self.nodes.get_with_flags(creator.0).1;
        if C::u8_load_relaxed(flags) & HAS_INDEX == 0 {
            return std::ptr::null();
        }
        let index = self.index_at(creator);
        index.with(|s| s.epoch += 1);
        ctl.children.set(std::ptr::from_ref(index).cast());
        index
    }

    /// Creator `id`'s index. The caller holds the creator's claim, and
    /// the index stays in `Inner::children` until the creator's release,
    /// which needs a delete claim that the compute claim excludes; the
    /// box does not move when the map does.
    pub(crate) fn index_at(&self, id: NodeId) -> &ChildIndex<C> {
        let index: *const ChildIndex<C> = &**self
            .inner
            .read()
            .children
            .get(&id.0)
            .expect("incr-core: HAS_INDEX is set on a node with no child index");
        // SAFETY: see the doc comment.
        unsafe { &*index }
    }

    /// What creator `id`'s landing does with its children, before its
    /// landing CAS while its claim is held. Everything a delete here
    /// would drop goes into `bag`, which drains after the landing.
    #[inline(always)]
    pub(crate) fn children_landing(
        &self,
        id: NodeId,
        flags: &C::U8,
        landing: ChildLanding,
        opened: bool,
        bag: &mut Deferred<C>,
    ) {
        if C::u8_load_relaxed(flags) & HAS_INDEX != 0 {
            self.land_children(id, landing, opened, bag);
        }
    }

    /// `opened` says the last run reached the index; one that did not
    /// touched no child, so a sweep after it numbers a run of its own.
    #[cold]
    #[inline(never)]
    pub(crate) fn land_children(
        &self,
        id: NodeId,
        landing: ChildLanding,
        opened: bool,
        bag: &mut Deferred<C>,
    ) {
        let index = self.index_at(id);
        let candidates = match landing {
            ChildLanding::Stashed => return,
            ChildLanding::Keep => {
                index.with(|s| s.land());
                return;
            }
            ChildLanding::Discard => {
                let keys = value_code::run(|| index.with(|s| s.unbind_created()));
                for k in keys {
                    bag.push_erased(k);
                }
                index.with(|s| {
                    s.land();
                    s.orphans().to_vec()
                })
            }
            ChildLanding::Sweep => index.with(|s| {
                if !opened {
                    s.epoch += 1;
                }
                s.land();
                s.sweep_candidates()
            }),
        };
        if candidates.is_empty() {
            return;
        }
        let deleted: HashSet<ErasedIncr> = self.sweep_units(id, &candidates, bag);
        if deleted.is_empty() {
            return;
        }
        let keys = value_code::run(|| index.with(|s| s.forget(&|h| deleted.contains(&h))));
        for k in keys {
            bag.push_erased(k);
        }
    }
}
