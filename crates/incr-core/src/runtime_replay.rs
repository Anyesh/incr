//! Deep chains by unwind and replay: the cold read paths a frame budget
//! sends reads down, the doom that releases the node a read wanted and
//! marks the frames below it, the discard of a doomed frame, and the
//! driver that runs a context's entries until its target lands.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use super::*;
use crate::cycle_state::Held;
use crate::dep_stack::Placement;
use crate::replay::{self, Delivered, Delivery, Entry, ReadKey, Retry, VFrame};
use crate::segmented_nodes::{NOT_REPLAYABLE, REPLAYABLE};

/// What the doom point saw, for tests.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct DoomSeen {
    pub(crate) node: u32,
    /// The node's state after its release.
    pub(crate) state: NodeState,
    pub(crate) verified_at: u64,
    /// Frames on the stack at the doom, read by a release-only test.
    #[cfg_attr(debug_assertions, allow(dead_code))]
    pub(crate) depth: usize,
}

/// An unwind a reading frame took from a pending delivery, for tests.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeliverySeen {
    pub(crate) frame: u32,
    pub(crate) read: u32,
    pub(crate) cycle: bool,
    pub(crate) kind: ReadKind,
}

/// The context a driver found when it picked an entry, for tests.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct EntrySeen {
    pub(crate) context: u64,
    pub(crate) level: u32,
    /// The front entry of each level, the target at level 0.
    pub(crate) fronts: Vec<Option<u32>>,
}

/// How a virtual frame's pop left its context's driver.
enum VPop {
    Next,
    /// A doom or a dropped chain pushed entries: the driver picks its next
    /// entry again.
    Restart,
    /// The run of the frame at the popped position ended, landing this
    /// node (`true`) or unwinding: no retry computes it again.
    Finished(u32, bool),
}

/// Places the frames a virtual frame's pop pushes above its position, and
/// restores the stack's bias when the pop ends, an unwind included.
struct Placed<'a, D: DepStack>(&'a D, Placement);

impl<D: DepStack> Drop for Placed<'_, D> {
    fn drop(&mut self) {
        self.0.restore_placement(self.1);
    }
}

/// A frame below a doomed read, as marking sees it.
struct Below {
    node: u32,
    cidx: u32,
    ctl: *const ComputeCtl,
    reading: u64,
    ordinal: u32,
    path: u64,
}

impl<C: Cells> Runtime<C> {
    /// Whether node `slot`'s compute may be doomed.
    pub(super) fn replayable(&self, slot: u32) -> bool {
        let bits = C::u8_load_relaxed(self.nodes.get_with_flags(slot).1);
        bits & NOT_REPLAYABLE == 0 && (bits & REPLAYABLE != 0 || self.all_replayable)
    }

    /// Whether the node a frame is blocked reading (`slot | generation <<
    /// 32`) has a type with a placeholder, so the frame can take one.
    fn reading_has_placeholder(&self, reading: u64) -> bool {
        let node = self.nodes.get(reading as u32);
        let tag = node.type_tag();
        C::fence_acquire();
        node.generation() == (reading >> 32) as u32 && self.arenas.erased(tag).has_placeholder()
    }

    /// Whether a read from frame `f` is at or past a budget.
    fn over_budget(&self, f: ActiveRead) -> bool {
        if f.depth >= self.max_frames.load(std::sync::atomic::Ordering::Relaxed) {
            return true;
        }
        let bytes = self
            .max_stack_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        if bytes == 0 {
            return false;
        }
        let here = 0_u8;
        let here = std::hint::black_box(std::ptr::from_ref(&here)) as usize;
        here.abs_diff(cycle_state::stack_base()) > bytes
    }

    /// A speculative entry's run lands only Clean values: a node it would
    /// land Dirty, poisoned or over a dep that is not Clean, has a history
    /// the unbounded run may never give it, and a later walk of its deps
    /// would differ from a first run of its closure. Such a pop goes on as
    /// a discard unwind.
    #[cold]
    #[inline(never)]
    pub(super) fn speculative_lands_dirty(
        &self,
        word: u64,
        ctl: &ComputeCtl,
        deps: &[(NodeId, u32)],
    ) -> bool {
        word & value_code::REPLAY_BIT != 0
            && replay::speculative()
            && (ctl.has(cycle::POISONED)
                || deps.iter().any(|&(d, _)| {
                    !matches!(
                        state::load::<C>(self.nodes.get(d.0).state_cell()),
                        NodeState::Clean | NodeState::Cyclic
                    )
                }))
    }

    /// A slow read that value code, a doomed or budgeted frame, or an open
    /// replay context sends off the hot path.
    #[cold]
    #[inline(never)]
    pub(super) fn read_cold<T: Value>(
        &self,
        handle: Incr<T>,
        observed: NodeState,
        kind: ReadKind,
        frame: Option<ActiveRead>,
        mut word: u64,
    ) -> Result<T, Cycle> {
        if word & value_code::COUNTER_MASK != 0 {
            return Ok(self.value_code_read(handle, self.nodes.get(handle.slot())));
        }
        // A read while a cycle's unwind kind is set comes from a closure that
        // caught that unwind: the kind is stale, and a panic of the node read
        // next would land as a cycle's discard instead of Failed.
        if word & value_code::UNWIND_MASK != 0
            && cycle_state::unwind_kind() == Some(UnwindKind::Cycle)
        {
            cycle_state::set_unwind_kind(None);
            word = value_code::thread_word();
        }
        // A delivery goes to its frame's retry in a non-speculative run: a
        // speculative run that computes the same node is on another path,
        // and a doomed frame is the run before the retry.
        let scope = (word & value_code::REPLAY_BIT != 0 && !replay::speculative()).then(|| {
            replay::ReadScope {
                rt: self.runtime_id,
                reading: u64::from(handle.slot()) | (u64::from(handle.generation()) << 32),
            }
        });
        if scope.is_some() {
            if let Some(read) = frame
                // SAFETY: the active frame's control block is live.
                .filter(|f| !unsafe { &*f.ctl }.has(cycle::DOOMED))
                .and_then(|f| self.take_delivery(f, handle, kind))
            {
                return read;
            }
        }
        let mut mode = 0;
        let mut _speculating = None;
        if let Some(f) = frame {
            // SAFETY: the active frame's control block is live.
            let ctl = unsafe { &*f.ctl };
            if ctl.has(cycle::DOOMED) {
                if let Some(value) = self.read_doomed(handle, f) {
                    return Ok(value);
                }
                // A node with a virtual frame stands for a frame below this
                // one and can't be computed in it. With no placeholder to
                // read, the doomed frame's run ends here: it is discarded
                // whatever it would go on to read.
                if word & value_code::REPLAY_BIT != 0
                    && !cycle_state::holds(self.runtime_id, handle.slot())
                    && self
                        .virtual_position(handle.slot(), handle.generation())
                        .is_some()
                {
                    self.drop_unwind();
                }
                // A doomed frame has read a placeholder, so the unbounded
                // run may never make this read: what it computes natively
                // is speculative, as an entry pushed here would be.
                _speculating = Some(replay::Speculating::enter());
            } else {
                if word & value_code::REPLAY_BIT != 0 {
                    if let Some(read) = self.judge_virtual(handle, f, kind) {
                        return read;
                    }
                }
                if self.over_budget(f)
                    && !ctl.has(cycle::NO_DOOM)
                    && self.replayable(f.node.0)
                    && self.placeholder::<T>().is_some()
                {
                    mode |= ARMED;
                }
            }
        }
        if word & value_code::REPLAY_BIT != 0
            && replay::with(|c| replay::innermost(c, self.runtime_id).is_some())
        {
            mode |= VIRT;
        }
        match mode {
            0 => self.read_walk::<T, 0>(handle, observed, kind, word, frame),
            ARMED => self.read_walk::<T, ARMED>(handle, observed, kind, word, frame),
            VIRT => self.read_walk::<T, VIRT>(handle, observed, kind, word, frame),
            _ => self.read_walk::<T, { ARMED | VIRT }>(handle, observed, kind, word, frame),
        }
    }

    /// Passes `low` to the parent of the frame at position `cidx`: the
    /// virtual frame below it when there is one (the frame is a driver's
    /// entry, or an episode a virtual frame's pop runs), else the active
    /// frame. Returns the parent's position and node, and whether it is a
    /// virtual frame.
    pub(super) fn lower_parent(&self, cidx: u32, low: u32) -> (u32, u32, Option<u64>) {
        let rt = self.runtime_id;
        if value_code::thread_word() & value_code::REPLAY_BIT != 0 && cidx > 0 {
            let below = cidx - 1;
            let lowered = replay::with(|c| {
                c.iter_mut()
                    .rev()
                    .filter(|x| x.rt == rt && below >= x.boundary)
                    .find_map(|x| x.vstack.get_mut((below - x.boundary) as usize))
                    .map(|v| {
                        v.low = v.low.min(low);
                        (v.node, v.path)
                    })
            });
            if let Some((node, path)) = lowered {
                return (below, node, Some(path));
            }
        }
        match self.dep_stack.active_frame(rt) {
            Some((parent, at)) => {
                // SAFETY: the active frame's control block is live.
                unsafe { &*parent }.lower(low);
                let node = self
                    .dep_stack
                    .active_compute(rt)
                    .map_or(u32::MAX, |(n, _)| n.0);
                (at, node, None)
            }
            None => (cidx.saturating_sub(1), u32::MAX, None),
        }
    }

    /// The highest virtual frame of `slot` at `generation` in a context of
    /// this runtime, as a stack position.
    pub(super) fn virtual_position(&self, slot: u32, generation: u32) -> Option<u32> {
        let rt = self.runtime_id;
        replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .filter_map(|x| x.position(slot, generation).map(|pos| x.boundary + pos))
                .max()
        })
    }

    /// The nodes of this runtime's virtual frames at stack position `p`
    /// or above, and the claims their runs had deferred at or above `p`.
    pub(super) fn virtual_slots_from(&self, p: u32) -> Vec<u32> {
        let rt = self.runtime_id;
        replay::with(|c| {
            let vframes = c
                .iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| x.vstack.iter());
            vframes
                .clone()
                .filter(|v| v.cidx >= p)
                .map(|v| v.node)
                .chain(
                    vframes
                        .flat_map(|v| v.deferred.iter())
                        .filter(|&&(_, anchor)| anchor >= p)
                        .map(|&(slot, _)| slot),
                )
                .collect()
        })
    }

    /// The claims of this runtime the frame at `cidx`, of `node`, at
    /// `path`, owns, as `(slot, anchor)`, which the virtual frame it
    /// becomes owns from now.
    fn owned_claims(&self, cidx: u32, node: u32, path: u64) -> Vec<(u32, u32)> {
        let rt = self.runtime_id;
        cycle_state::with_held(|h| {
            h.iter_mut()
                .filter(|((r, _), e)| {
                    *r == rt
                        && e.owner == cidx
                        && (e.owner_node == node || e.owner_node == u32::MAX)
                        && e.owner_virtual.is_none_or(|p| p == path)
                })
                .map(|(&(_, slot), e)| {
                    e.own(cidx, node, Some(path));
                    (slot, e.anchor)
                })
                .collect()
        })
    }

    /// The judgement of a read by frame `f`, not doomed, of a node that is
    /// not Clean: `None` unless the node has a virtual frame this thread
    /// does not hold. Such a read never computes the node, which stands
    /// for a frame below `f` in the unbounded run. A suspect edge between
    /// that frame and `f` drops the chain above it; otherwise the read is a
    /// virtual hit, which claims the virtual frames' nodes from there up
    /// and is served by the read rules as a read of a frame on the stack.
    fn judge_virtual<T: Value>(
        &self,
        handle: Incr<T>,
        f: ActiveRead,
        kind: ReadKind,
    ) -> Option<Result<T, Cycle>> {
        let rt = self.runtime_id;
        let slot = handle.slot();
        // A member run sits at its episode's anchor but reads from the top
        // of the stack, above every frame between.
        let top = f.cidx.max(self.dep_stack.next_cidx().wrapping_sub(1));
        let mut spins = 0;
        loop {
            match state::load::<C>(self.nodes.get(slot).state_cell()) {
                NodeState::New
                | NodeState::Dirty
                | NodeState::Computing
                | NodeState::ComputingDirty => {}
                _ => return None,
            }
            // A doomed frame of the node is still on the stack, under a
            // native compute its run makes: the read is one of that frame.
            if cycle_state::holds(rt, slot) || self.dep_stack.frame_cidx(rt, NodeId(slot)).is_some()
            {
                return None;
            }
            let p = self.virtual_position(slot, handle.generation())?;
            // Before its loop runs, a context's virtual frames stand for
            // doomed frames that have popped, so they are on no stack, as
            // are those a doom in the running entry's run made at or above
            // its level. Only speculative runs under the doomed frames read
            // them.
            if !replay::with(|c| {
                c.iter().any(|x| {
                    x.rt == rt
                        && x.position(slot, handle.generation()).is_some_and(|q| {
                            x.boundary + q == p && x.running.is_some_and(|l| q < l)
                        })
                })
            }) {
                debug_assert!(
                    replay::speculative(),
                    "incr-core: only a speculative run reads before a loop runs"
                );
                self.drop_unwind();
            }
            if self.suspect_above(p, top) {
                debug_assert!(
                    replay::speculative(),
                    "incr-core: a suspect edge lies under a speculative entry"
                );
                self.drop_unwind();
            }
            match self.claim_virtual(p, top) {
                Ok(()) => {
                    return Some(
                        self.cycle_read(handle, NodeState::Computing, kind)
                            .expect("incr-core: a virtual hit holds the node it read"),
                    );
                }
                Err(on) => loop {
                    spins += 1;
                    self.wait_on_claim(on, spins);
                    if !is_computing(state::load::<C>(self.nodes.get(on.0).state_cell())) {
                        break;
                    }
                },
            }
        }
    }

    /// Whether an edge into a frame above position `p`, up to `top`, is
    /// suspect: a virtual frame's, or a running entry's.
    fn suspect_above(&self, p: u32, top: u32) -> bool {
        let rt = self.runtime_id;
        replay::with(|c| {
            c.iter().filter(|x| x.rt == rt).any(|x| {
                x.vstack
                    .iter()
                    .any(|v| v.cidx > p && v.cidx <= top && v.suspect)
                    || x.running.is_some_and(|level| {
                        let at = x.boundary + level;
                        at > p && at <= top && x.running_entry.is_some_and(|e| e.suspect)
                    })
            })
        })
    }

    /// A speculative run reached a group verify, whose member runs and
    /// landings would outlive its discard and change what the native retry
    /// finds: the claim lands back as it was and the chain is dropped, so
    /// that the retry verifies the group.
    #[cold]
    #[inline(never)]
    pub(super) fn drop_speculative_verify(&self, node: &NodeData<C>, claimed_dirty: bool) -> ! {
        let target = if claimed_dirty {
            NodeState::Dirty
        } else {
            node.set_verified_at(0);
            NodeState::New
        };
        self.land(node, target, target);
        self.drop_unwind()
    }

    /// Raises the discard unwind that drops a speculative chain: every frame
    /// it passes takes the discard landing, and the driver's catch of the
    /// speculative entry drops the chain.
    pub(super) fn drop_unwind(&self) -> ! {
        cycle_state::set_unwind_kind(Some(UnwindKind::Discard));
        resume_unwind(Box::new(cycle::DiscardUnwind))
    }

    /// A virtual hit's claims: the node of every virtual frame from
    /// position `p` up to `top` that this thread does not hold, as held
    /// claims anchored at `p`. On a claim another thread holds, the claims
    /// taken are rolled back with the discard landing and that node is
    /// returned for the read to wait on.
    fn claim_virtual(&self, p: u32, top: u32) -> Result<(), NodeId> {
        let rt = self.runtime_id;
        let mut frames: Vec<(u32, u32, u32)> = replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| x.vstack.iter())
                .filter(|v| v.cidx >= p && v.cidx <= top)
                .map(|v| (v.cidx, v.node, v.generation))
                .collect()
        });
        frames.sort_unstable();
        let mut taken: Vec<(u32, bool)> = Vec::new();
        for (_, slot, generation) in frames {
            // A doomed frame still running holds its node's claim itself.
            if cycle_state::holds(rt, slot) || self.dep_stack.frame_cidx(rt, NodeId(slot)).is_some()
            {
                continue;
            }
            let (node, flags) = self.nodes.get_with_flags(slot);
            let claimed = state::try_claim_compute::<C>(node.state_cell());
            let first = match claimed {
                Ok(src) if node.generation() == generation => src == NodeState::New,
                Ok(src) => {
                    self.land(node, src, src);
                    self.roll_back_hit(&taken);
                    return Err(NodeId(slot));
                }
                Err(_) => {
                    self.roll_back_hit(&taken);
                    return Err(NodeId(slot));
                }
            };
            let seeded = C::u8_load_relaxed(flags) & HAS_STRATEGY != 0;
            cycle_state::with_held(|h| {
                h.insert((rt, slot), Held::new(generation, p, seeded, first));
            });
            replay::with(|c| {
                if let Some(x) = c
                    .iter_mut()
                    .rev()
                    .find(|x| x.rt == rt && x.running_entry.is_some())
                {
                    let level = x.running.expect("a context with an entry is running");
                    if p < x.boundary + level {
                        x.claims.push((level, slot));
                    }
                }
            });
            taken.push((slot, first));
        }
        Ok(())
    }

    fn roll_back_hit(&self, taken: &[(u32, bool)]) {
        let rt = self.runtime_id;
        let stamp_rev = self.current_revision();
        let mut bag = Deferred::new();
        for &(slot, first) in taken {
            cycle_state::with_held(|h| h.remove(&(rt, slot)));
            self.discard_land(NodeId(slot), first, false, stamp_rev, &mut bag);
        }
        bag.drain(self);
    }

    /// A read made in a doomed frame: the placeholder, after pushing the
    /// node as an entry above the frame unless this thread computes,
    /// holds or virtually computes it. `None` for a type with no
    /// placeholder, which the read computes natively.
    fn read_doomed<T: Value>(&self, handle: Incr<T>, f: ActiveRead) -> Option<T> {
        let placeholder = self.placeholder::<T>()?;
        let rt = self.runtime_id;
        let slot = handle.slot();
        if state::load::<C>(self.nodes.get(slot).state_cell()) == NodeState::Clean {
            return None;
        }
        // SAFETY: the active frame's control block is live.
        let ctl = unsafe { &*f.ctl };
        let reached =
            self.dep_stack.frame_cidx(rt, NodeId(slot)).is_some() || cycle_state::holds(rt, slot);
        let suspect = ctl.has(cycle::PH_SEEN);
        replay::with(|c| {
            if reached
                || c.iter()
                    .any(|x| x.rt == rt && x.position(slot, handle.generation()).is_some())
            {
                return;
            }
            let at = replay::covering(c, rt, f.cidx)
                .expect("incr-core: a doomed frame lies in a replay context");
            let x = &mut c[at];
            let spec = x.vstack[(f.cidx - x.boundary) as usize].spec;
            x.push_entry(
                f.cidx + 1 - x.boundary,
                Entry {
                    node: slot,
                    generation: handle.generation(),
                    suspect,
                    spec: spec || suspect,
                    retry: Retry::Walk,
                },
            );
        });
        ctl.raise(cycle::PH_SEEN);
        Some(placeholder)
    }

    /// The answer to a read whose walk was doomed or discarded: the reading
    /// frame is doomed and gets its type's placeholder.
    #[cold]
    #[inline(never)]
    pub(super) fn placeholder_read<T: Value>(&self, frame: Option<ActiveRead>) -> T {
        let f = frame.expect("incr-core: only a frame's read is answered by a placeholder");
        // SAFETY: the active frame's control block is live.
        let ctl = unsafe { &*f.ctl };
        debug_assert!(
            ctl.has(cycle::DOOMED),
            "incr-core: a placeholder read dooms its frame"
        );
        ctl.raise(cycle::PH_SEEN);
        self.placeholder::<T>()
            .expect("incr-core: a doomed read's type has a placeholder")
    }

    /// The doom of the read that wanted `id`, which this thread claimed
    /// (from Dirty if `claimed_dirty`, else from New) and whose closure the
    /// read would run past the budget: the claim is released with the
    /// discard landing, keeping a group member's record, the frames below
    /// that can take a placeholder are marked, and `id` becomes an entry.
    #[cold]
    #[inline(never)]
    pub(super) fn doom(
        &self,
        id: NodeId,
        node: &NodeData<C>,
        generation: u32,
        claimed_dirty: bool,
        direct: bool,
    ) -> Computed {
        let target = if claimed_dirty {
            NodeState::Dirty
        } else {
            node.set_verified_at(0);
            NodeState::New
        };
        self.land(node, target, target);
        #[cfg(test)]
        self.dooms.lock().unwrap().push(DoomSeen {
            node: id.0,
            state: node.state(),
            verified_at: node.verified_at(),
            depth: self.dep_stack.depth(),
        });
        self.mark(
            id,
            generation,
            if direct {
                Retry::Direct
            } else {
                Retry::Compute
            },
        );
        self.park(ParkPoint::AfterDoom(id));
        Computed::Doomed
    }

    /// A clone of node `slot`'s value, if its slot holds one.
    fn slot_value(&self, slot: u32) -> Option<cycle::Erased> {
        let node = self.nodes.get(slot);
        self.arenas
            .erased(node.type_tag())
            .read_erased(node.arena_slot())
    }

    /// The earlier reads of the frames being marked that returned a value
    /// a landing had left Dirty. A retry's read gets that value again, as
    /// the frame's first run did, where a fresh read would walk the Dirty
    /// node, or read what a later landing replaced it with.
    fn landed_reads(&self, marked: &[Below], vframes: &[VFrame]) -> Vec<(ReadKey, cycle::Erased)> {
        let rt = self.runtime_id;
        let mut out = Vec::new();
        for f in marked {
            // SAFETY: a frame's control block is live while it is on the
            // stack, and these frames are being marked.
            if !unsafe { &*f.ctl }.has(cycle::DIRTY_READ) {
                continue;
            }
            let Some(v) = vframes.iter().find(|v| v.cidx == f.cidx) else {
                continue;
            };
            for read in replay::take_dirty_reads(rt, f.cidx, f.ctl) {
                if vframes
                    .iter()
                    .any(|w| u64::from(w.node) == read.reading & 0xffff_ffff)
                {
                    continue;
                }
                let key = ReadKey {
                    reading: read.reading,
                    ordinal: read.ordinal,
                    ..ReadKey::blocked(v)
                };
                out.push((key, read.value));
            }
        }
        out
    }

    /// A read by frame `f` returned `value` from a node a landing left
    /// Dirty. If a budget is set, the frame may be doomed and retried, and
    /// the retry's read must get this value again.
    #[cold]
    #[inline(never)]
    pub(super) fn note_dirty_read<T: Value>(&self, f: ActiveRead, handle: Incr<T>, value: &T) {
        if self.max_frames.load(std::sync::atomic::Ordering::Relaxed) == u32::MAX
            && self
                .max_stack_bytes
                .load(std::sync::atomic::Ordering::Relaxed)
                == 0
        {
            return;
        }
        // SAFETY: the active frame's control block is live.
        let ctl = unsafe { &*f.ctl };
        let fresh = !ctl.has(cycle::DIRTY_READ);
        ctl.raise(cycle::DIRTY_READ);
        let rt = self.runtime_id;
        let value: cycle::Erased = Box::new(value_code::run(|| value.clone()));
        replay::note_dirty_read(
            replay::DirtyRead {
                rt,
                cidx: f.cidx,
                ctl: f.ctl,
                reading: u64::from(handle.slot()) | (u64::from(handle.generation()) << 32),
                ordinal: self.dep_stack.reads_of_current(rt),
                value,
            },
            fresh,
        );
    }

    /// Marks the reading frame F and the frames below it that can be
    /// doomed, makes them virtual frames, and pushes `n` one level above F.
    fn mark(&self, n: NodeId, generation: u32, retry: Retry) {
        let rt = self.runtime_id;
        let mut frames: Vec<Below> = self.dep_stack.with(|s| {
            s.frames
                .iter()
                .rev()
                .take_while(|f| f.rt == rt)
                .map(|f| Below {
                    node: f.node.0,
                    cidx: f.cidx,
                    ctl: f.ctl,
                    reading: f.reading,
                    ordinal: f
                        .deps
                        .iter()
                        .filter(|&&(d, g)| (u64::from(d.0) | (u64::from(g) << 32)) == f.reading)
                        .count() as u32,
                    path: 0,
                })
                .collect()
        });
        // Bottom up, each frame's path extends the one below it.
        let mut below: Option<(u32, u64, (u64, u32))> = None;
        for f in frames.iter_mut().rev() {
            f.path = match below {
                Some((cidx, path, step)) if cidx + 1 == f.cidx => replay::extend_path(path, step),
                _ => self.path_at(f.cidx),
            };
            below = Some((f.cidx, f.path, (f.reading, f.ordinal)));
        }
        let top = frames
            .first()
            .expect("incr-core: a doom has a reading frame");
        let (running, floor, running_entry, base_spec) =
            replay::with(|c| match replay::covering(c, rt, top.cidx) {
                Some(at) => {
                    let x = &c[at];
                    match x.running {
                        Some(level) => {
                            (Some(x.id), x.boundary + level, x.running_entry, x.spec_base)
                        }
                        None => (None, x.boundary, None, x.spec_base),
                    }
                }
                None => (None, 0, None, false),
            });
        let mut count = 1;
        for f in &frames[1..] {
            // SAFETY: a frame's control block is live while it is on the
            // stack, and these frames are below the reading one.
            let ctl = unsafe { &*f.ctl };
            if f.cidx < floor
                || ctl.flags.get() & (cycle::DOOMED | cycle::NO_DOOM) != 0
                || !self.replayable(f.node)
                || !self.reading_has_placeholder(f.reading)
            {
                break;
            }
            count += 1;
        }
        let marked = &frames[..count];
        let bottom = &marked[count - 1];
        let joined = running.filter(|_| bottom.cidx == floor);
        let spec = replay::speculative()
            || match (joined, running_entry) {
                (Some(_), Some(e)) => e.spec,
                _ => running_entry.map_or(base_spec, |e| e.spec),
            };
        // The compute whose discard runs the context is the bottom frame's,
        // whose retry goes straight to the closure as that frame had, or
        // for a member run, that of the episode's root: its pop or its
        // group verify ran the episode, and runs it again.
        // SAFETY: as above.
        let (target, target_retry) = if unsafe { &*bottom.ctl }.has(cycle::BUFFERED) {
            (
                replay::member_run_root(bottom.cidx).unwrap_or(bottom.node),
                Retry::Compute,
            )
        } else {
            (bottom.node, Retry::Direct)
        };
        let id = match joined {
            Some(id) => id,
            None => replay::open(
                rt,
                bottom.cidx,
                Entry {
                    node: target,
                    generation: self.nodes.get(target).generation(),
                    suspect: false,
                    spec,
                    retry: target_retry,
                },
            ),
        };
        let vframes: Vec<VFrame> = (0..count)
            .rev()
            .map(|i| {
                let f = &marked[i];
                // SAFETY: as above.
                let ctl = unsafe { &*f.ctl };
                let suspect = if i + 1 == count {
                    joined.is_some() && running_entry.is_some_and(|e| e.suspect)
                } else {
                    // SAFETY: as above.
                    unsafe { &*marked[i + 1].ctl }.has(cycle::PH_SEEN)
                };
                VFrame {
                    node: f.node,
                    generation: self.nodes.get(f.node).generation(),
                    cidx: f.cidx,
                    low: ctl.lowlink.get().min(f.cidx),
                    suspect,
                    spec,
                    reading: f.reading,
                    ordinal: f.ordinal,
                    path: f.path,
                    maker: 0,
                    ep: if ctl.has(cycle::BUFFERED) {
                        replay::member_run_episode(f.cidx)
                    } else {
                        None
                    },
                    deferred: if ctl.has(cycle::BUFFERED) {
                        Vec::new()
                    } else {
                        self.owned_claims(f.cidx, f.node, f.path)
                    },
                }
            })
            .collect();
        let landed = self.landed_reads(marked, &vframes);
        // SAFETY: as above.
        let top_ctl = unsafe { &*top.ctl };
        let suspect = top_ctl.has(cycle::PH_SEEN);
        replay::with(|c| {
            let x = c
                .iter_mut()
                .find(|x| x.id == id)
                .expect("incr-core: the doom's context is open");
            let maker = bottom.cidx - x.boundary;
            // The running entry's run got past its verify into the closure
            // that this doom discards: its next retry goes straight to the
            // closure, as the unbounded run verifies once, before deps that
            // the entries above land can change the verify's answer.
            // SAFETY: as above.
            if joined.is_some()
                && !unsafe { &*bottom.ctl }.has(cycle::BUFFERED)
                && running_entry.is_some_and(|e| e.node == bottom.node)
            {
                if let Some(e) = x
                    .levels
                    .get_mut(maker as usize)
                    .and_then(|q| q.front_mut())
                    .filter(|e| e.node == bottom.node)
                {
                    e.retry = Retry::Direct;
                }
            }
            for mut v in vframes {
                v.maker = maker;
                x.push_vframe(v);
            }
            for (key, value) in landed {
                // A frame marked again keeps the delivery its first mark, or
                // the entry it was blocked on, made for the read.
                if !x.pending.iter().any(|d| {
                    matches!(d.what, Delivered::Landed(_) | Delivered::Unwind { .. })
                        && d.key == key
                }) {
                    x.pending.push(Delivery {
                        key,
                        maker,
                        taken: false,
                        blocked: false,
                        what: Delivered::Landed(value),
                    });
                }
            }
            #[cfg(test)]
            assert!(
                x.position(n.0, generation).is_none(),
                "incr-core: a doom pushed node {} which has a virtual frame",
                n.0
            );
            x.push_entry(
                top.cidx + 1 - x.boundary,
                Entry {
                    node: n.0,
                    generation,
                    suspect,
                    spec: spec || suspect,
                    retry,
                },
            );
        });
        for f in marked {
            // SAFETY: as above.
            unsafe { &*f.ctl }.raise(cycle::DOOMED);
        }
        top_ctl.raise(cycle::PH_SEEN);
        if value_code::thread_word() & value_code::MEMO_BIT != 0 {
            let parked = taint_memo::take_from(bottom.cidx + 1);
            replay::with(|c| {
                if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                    x.memo.extend(parked);
                }
            });
        }
    }

    /// The pop of a doomed frame: its run, its panic included, lands
    /// nothing, and every claim held at or above it takes the discard
    /// landing, except those virtual frames own, whose root stays held.
    #[cold]
    #[inline(never)]
    pub(super) fn discard_doomed(
        &self,
        id: NodeId,
        attempt: Attempt,
        ctl: &ComputeCtl,
        payload: Option<Box<dyn std::any::Any + Send>>,
        bag: &mut Deferred<C>,
    ) -> Computed {
        // An episode a read in the run suspended waits for its member's
        // virtual frame, not for this frame's retry.
        let mut keep = replay::suspended_members(self.runtime_id);
        let rooted = self.keep_virtual(id, attempt.cidx, &mut keep);
        self.release_held_except(attempt.cidx, true, &keep, bag);
        if let Some(value) = ctl.buf.take() {
            bag.push_erased(value);
        }
        if let Some(payload) = payload {
            bag.push_erased(payload);
        }
        let wrote = ctl.wrote.get() != 0;
        match rooted {
            Some(path) => self.hold_doomed_root(id, attempt, wrote, path, bag),
            // A replayable closure is pure, so its retry asks for the same
            // children, which the entries above compute meanwhile; unbinding
            // them would make the retry create them New and doom again.
            None => self.discard_land_with(
                id,
                attempt.first,
                wrote,
                attempt.stamp_rev,
                ChildLanding::Keep,
                bag,
            ),
        }
        replay::unwind_past(self.runtime_id, attempt.cidx + 1);
        Computed::Discarded
    }

    /// The claims that virtual frames at or above `cidx` own since the
    /// mark, added to `keep`: the unbounded run's frames are still on the
    /// stack with them held while the blocked read runs, so the entries
    /// above read them held, not New. When some are anchored at `cidx`, the
    /// doomed run of `id` there roots them, and the path of its virtual
    /// frame is returned.
    fn keep_virtual(&self, id: NodeId, cidx: u32, keep: &mut Vec<u32>) -> Option<u64> {
        let rt = self.runtime_id;
        let mut rooted = false;
        cycle_state::with_held(|h| {
            for (&(r, slot), e) in h.iter() {
                if r == rt && e.owner_virtual.is_some() && e.anchor >= cidx && e.owner >= cidx {
                    keep.push(slot);
                    rooted |= e.anchor == cidx;
                }
            }
        });
        if !rooted || cycle_state::holds(rt, id.0) {
            return None;
        }
        replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| x.vstack.iter())
                .find(|v| v.cidx == cidx && v.node == id.0)
                .map(|v| v.path)
        })
    }

    /// The doomed root of kept claims stays claimed, as a virtual hit's
    /// claim of its frame, unbuffered: its virtual frame's pop finishes its
    /// run and roots the episode, as the unbounded run's frame does.
    fn hold_doomed_root(
        &self,
        id: NodeId,
        attempt: Attempt,
        wrote: bool,
        path: u64,
        bag: &mut Deferred<C>,
    ) {
        let rt = self.runtime_id;
        let (node, flags) = self.nodes.get_with_flags(id.0);
        self.children_landing(id, flags, ChildLanding::Keep, false, bag);
        let seeded = C::u8_load_relaxed(flags) & HAS_STRATEGY != 0;
        let mut held = Held::new(node.generation(), attempt.cidx, seeded, attempt.first);
        held.own(attempt.cidx, id.0, Some(path));
        held.wrote = wrote;
        cycle_state::with_held(|h| h.insert((rt, id.0), held));
    }

    /// A walk's compute came back discarded. If the doom opened a context
    /// at the position that compute ran at, the walk is that context's
    /// opener and runs its loop here, then goes on as if the compute had
    /// landed; otherwise (the reading frame is doomed, or the walk is a
    /// driver entry's) the walk stops.
    #[cold]
    #[inline(never)]
    pub(super) fn walk_discarded(&self, visit: Visit) -> Option<Computed> {
        let cidx = self.dep_stack.next_cidx();
        let rt = self.runtime_id;
        let id = replay::with(|c| {
            let at = replay::innermost(c, rt)?;
            let x = &c[at];
            (x.running.is_none() && x.boundary == cidx).then_some(x.id)
        })?;
        Some(self.run_loop(id, visit))
    }

    /// The driver of context `id`: runs its entries, highest level first
    /// and oldest first within a level, each at its level's position,
    /// until the level-0 target lands, and returns how the target's last
    /// run left it.
    fn run_loop(&self, id: u64, visit: Visit) -> Computed {
        'entries: loop {
            let (level, entry, parked) = replay::with(|c| {
                let x = c
                    .iter_mut()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                let parked = std::mem::take(&mut x.memo);
                let (level, entry) = x
                    .next_entry()
                    .expect("incr-core: a context keeps its target until it lands");
                #[cfg(test)]
                self.entries.lock().unwrap().push(EntrySeen {
                    context: x.id,
                    level,
                    fronts: x.levels.iter().map(|q| q.front().map(|e| e.node)).collect(),
                });
                // Running, with no entry yet while its virtual frames pop.
                x.running = Some(level);
                x.running_entry = None;
                (level, entry, parked)
            });
            if !parked.is_empty() {
                taint_memo::restore(parked);
            }
            // The virtual frames at and above the entry's level pop, highest
            // first, each as the frame it stands for would.
            let mut popped_frames: Vec<(VFrame, bool)> = Vec::new();
            loop {
                let popped = replay::with(|c| {
                    let x = c
                        .iter_mut()
                        .find(|x| x.id == id)
                        .expect("incr-core: a running context is open");
                    if x.vstack.len() > level as usize {
                        x.pop_vframe().map(|v| (v, !x.vstack.is_empty()))
                    } else {
                        None
                    }
                });
                let Some((v, virtual_below)) = popped else {
                    break;
                };
                self.note_ancestor(&v);
                // A stand-in run in the pop finishes the frame that was
                // reading the frame popped just above, so it gets the direct
                // compute of that read an entry's retry gets below.
                if let Some((u, false)) = popped_frames.last() {
                    let read = u64::from(u.node) | (u64::from(u.generation) << 32);
                    if u.cidx == v.cidx + 1 && v.reading == read {
                        replay::with(|c| {
                            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                                x.deliver_direct(&v, level);
                            }
                        });
                    }
                }
                let popped = {
                    let _placed = Placed(
                        &self.dep_stack,
                        self.dep_stack.place_next_at(v.cidx, virtual_below),
                    );
                    self.tarjan_pop(id, &v)
                };
                match popped {
                    VPop::Next => popped_frames.push((v.clone(), false)),
                    VPop::Restart => continue 'entries,
                    VPop::Finished(node, landed) => {
                        popped_frames.push((v.clone(), landed));
                        if let Some(computed) =
                            self.frame_finished(id, level, entry, &v, node, landed)
                        {
                            return computed;
                        }
                        continue 'entries;
                    }
                }
            }
            let boundary = replay::with(|c| {
                let x = c
                    .iter_mut()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                x.running = Some(level);
                x.running_entry = Some(entry);
                x.boundary
            });
            // An entry whose node this thread holds is a cycle member that a
            // virtual hit claimed after the entry was pushed, or that an R5
            // read doomed. Without a buffer it runs as a buffered member run,
            // which finishes the run its frame would have; with one it is
            // done.
            let held = cycle_state::with_held(|h| {
                h.get(&(self.runtime_id, entry.node))
                    .filter(|e| e.generation == entry.generation)
                    .map(|e| e.buf.is_some())
            });
            if let Some(buffered) = held {
                if !buffered {
                    let cidx = boundary + level;
                    self.retry_deliveries(id, entry.node, entry.generation, cidx);
                    let ran = {
                        let _speculating = entry.spec.then(replay::Speculating::enter);
                        self.member_run(entry.node, cidx, true)
                    };
                    replay::run_ended(self.runtime_id, cidx, matches!(ran, Ok(None)));
                    if !matches!(ran, Ok(None)) {
                        self.end_deliveries(id, entry.node, entry.generation, cidx);
                    }
                    match ran {
                        Ok(Some(low)) if low < cidx => self.pass_down(cidx, low),
                        Ok(Some(_)) => {}
                        // A doom discarded the run: its entries run first,
                        // and this entry stays for the retry.
                        Ok(None) => continue,
                        Err(payload) => {
                            let mut bag = Deferred::new();
                            self.continuation_unwound(entry.node, cidx, &mut bag);
                            bag.drain(self);
                            match self.entry_unwound(id, level, entry, boundary, payload) {
                                Some(computed) => return computed,
                                None => continue,
                            }
                        }
                    }
                }
                replay::with(|c| {
                    if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                        x.pop_entry(level);
                        x.claims.retain(|&(l, _)| l < level);
                    }
                });
                if level == 0 {
                    self.close_context(id, boundary);
                    return Computed::Deferred;
                }
                continue;
            }
            // The entry is the retry of the doomed run that its virtual frame
            // stood for, so it owns what that run deferred.
            let at = boundary + level;
            self.own_deferred(at, entry.node);
            for (v, _) in popped_frames
                .iter()
                .filter(|&(v, landed)| !landed && v.cidx > at)
            {
                let read = u64::from(v.node) | (u64::from(v.generation) << 32);
                if let Some((w, _)) = popped_frames
                    .iter()
                    .find(|(w, _)| w.cidx + 1 == v.cidx && w.reading == read)
                {
                    replay::with(|c| {
                        if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                            x.deliver_direct(w, level);
                        }
                    });
                }
            }
            let placement = self.dep_stack.place_next_at(at, level > 0);
            let ran = {
                let _speculating = entry.spec.then(replay::Speculating::enter);
                catch_unwind(AssertUnwindSafe(|| {
                    if level == 0 {
                        // The target is the walk entry whose compute was
                        // discarded, and the walk has cleaned its deps, so
                        // it computes as that entry would, not as a new
                        // walk's root.
                        let visit = if entry.retry == Retry::Direct {
                            Visit::Direct
                        } else {
                            visit
                        };
                        self.compute_one::<VIRT>(NodeId(entry.node), visit, Some(entry.generation))
                    } else {
                        let (id, generation) = (NodeId(entry.node), Some(entry.generation));
                        match entry.retry {
                            Retry::Walk => self.ensure_clean_virt(id, entry.generation),
                            Retry::Compute => self.compute_one::<VIRT>(id, Visit::Full, generation),
                            Retry::Direct => {
                                self.compute_one::<VIRT>(id, Visit::Direct, generation)
                            }
                        }
                    }
                }))
            };
            self.dep_stack.restore_placement(placement);
            // A delivery its frame's retry never read is dropped once the
            // run ends, as the unbounded run drops it when a set changed
            // that frame's earlier reads. A run doomed again has not
            // ended: the frame comes back as an entry of its own, and its
            // next retry reads the delivery again.
            if matches!(ran, Ok(Computed::Discarded)) {
                replay::with(|c| {
                    if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                        for d in x.pending.iter_mut().filter(|d| d.maker == level) {
                            d.taken = false;
                        }
                    }
                });
            } else {
                let unread: Vec<Delivery> = replay::with(|c| {
                    c.iter_mut()
                        .find(|x| x.id == id)
                        .map(|x| x.pending.extract_if(.., |d| d.maker == level).collect())
                        .unwrap_or_default()
                });
                drop(unread);
            }
            let computed = match ran {
                Ok(computed) => computed,
                Err(payload) => match self.entry_unwound(id, level, entry, boundary, payload) {
                    Some(computed) => computed,
                    None => continue,
                },
            };
            if computed == Computed::Discarded {
                continue;
            }
            debug_assert_ne!(computed, Computed::Doomed);
            if computed == Computed::Done && level > 0 {
                self.hold_landing(id, level, entry.node, entry.generation);
            }
            replay::with(|c| {
                if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                    x.pop_entry(level);
                    // The claims the run's hits took below its level stay
                    // held for the episode their position runs.
                    x.claims.retain(|&(l, _)| l < level);
                }
            });
            if level == 0 {
                self.close_context(id, boundary);
                return computed;
            }
        }
    }

    /// A virtual frame's pop, as its frame's pop would be: nothing if its
    /// node has landed or been replaced since, a lowlink below its position
    /// passed down, and at a lowlink equal to its position with claims
    /// anchored there, the episode it roots, run here as a task. A
    /// speculative frame never runs one: its chain is dropped instead, and
    /// the driver picks its next entry again (`true`).
    fn tarjan_pop(&self, id: u64, v: &VFrame) -> VPop {
        let rt = self.runtime_id;
        let node = self.nodes.get(v.node);
        if node.generation() != v.generation {
            return VPop::Next;
        }
        match state::load::<C>(node.state_cell()) {
            NodeState::New | NodeState::Dirty => {}
            NodeState::Computing | NodeState::ComputingDirty if cycle_state::holds(rt, v.node) => {}
            _ => return VPop::Next,
        }
        // What the pop runs sits at the frame's position: a doom in it joins
        // the context there.
        replay::with(|c| {
            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                x.running = Some(v.cidx - x.boundary);
                x.running_entry = None;
            }
        });
        if let Some(ep) = v.ep {
            let ep = replay::take_episode(id, ep)
                .expect("incr-core: a member run's virtual frame keeps its episode");
            #[cfg(test)]
            {
                let queued_above = replay::with(|c| {
                    c.iter().find(|x| x.id == id).map_or(0, |x| {
                        x.levels
                            .iter()
                            .skip((v.cidx - x.boundary) as usize + 1)
                            .map(|q| q.len())
                            .sum()
                    })
                });
                self.episodes_seen
                    .lock()
                    .unwrap()
                    .push(cycle_impl::EpisodeSeen::of(&ep, Some(queued_above)));
            }
            let root = ep.root;
            let mut bag = Deferred::new();
            let end = catch_unwind(AssertUnwindSafe(|| self.run_episode(ep, &mut bag)));
            return match end {
                Ok(end) => self.episode_task_end(root, end, bag),
                Err(payload) => self.vframe_unwound(id, v, root.0, false, payload),
            };
        }
        let mut low = v.low;
        let unbuffered = cycle_state::with_held(|h| h.get(&(rt, v.node)).map(|e| e.buf.is_none()));
        if low < v.cidx && unbuffered != Some(false) {
            replay::with(|c| {
                if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                    x.lows.push((v.node, v.generation, v.cidx, v.path, low));
                }
            });
        }
        // A virtual hit claimed the node, so no retry recomputes it: its
        // frame's run finishes here, as the unbounded run's frame does
        // after its blocked read returns, and buffers its value. The claim
        // stands for that frame on the stack, so the frame's pop decides
        // what becomes of it, as a compute's pop would.
        let mut stand_in = None;
        if unbuffered == Some(true) {
            stand_in = Some((
                self.current_revision(),
                C::walked(&self.walk, &self.revision),
            ));
            self.retry_deliveries(id, v.node, v.generation, v.cidx);
            // The run finishes the frame's doomed run, so it owns what that
            // run deferred.
            self.own_deferred(v.cidx, v.node);
            let ran = {
                let _speculating = v.spec.then(replay::Speculating::enter);
                self.member_run(v.node, v.cidx, true)
            };
            replay::run_ended(rt, v.cidx, matches!(ran, Ok(None)));
            match ran {
                Ok(Some(l)) => {
                    self.end_deliveries(id, v.node, v.generation, v.cidx);
                    // The run started from the doomed runs' lowlink
                    // (`take_low`), so its own is the frame's.
                    low = l;
                }
                Ok(None) => return VPop::Restart,
                Err(payload) => {
                    self.end_deliveries(id, v.node, v.generation, v.cidx);
                    return self.vframe_unwound(id, v, v.node, true, payload);
                }
            }
        }
        if low < v.cidx {
            if stand_in.is_some() {
                // The run finished the frame, which defers as `cycle_pop`
                // defers a real one.
                cycle_state::with_held(|h| {
                    let e = h
                        .get_mut(&(rt, v.node))
                        .expect("incr-core: a stand-in is held");
                    e.anchor = v.cidx;
                    e.own(v.cidx, v.node, None);
                });
                self.pass_down(v.cidx, low);
            } else if unbuffered.is_some() {
                self.lower_parent(v.cidx, low);
            }
            // Otherwise the frame's run is not over: its entry, or the run
            // that computes it natively, retries it at this position and
            // passes its own lowlink down, or unwinds and passes none.
            return VPop::Next;
        }
        let rooted = cycle_state::with_held(|h| {
            h.iter().any(|(&(r, slot), e)| {
                r == rt && e.anchor >= v.cidx && (stand_in.is_none() || slot != v.node)
            }) || stand_in.is_some() && h[&(rt, v.node)].last.iter().any(|&(d, _)| d.0 == v.node)
        });
        if let (Some(revs), false) = (stand_in, rooted) {
            if !(v.spec && self.alone_lands_dirty(v.node)) {
                let mut bag = Deferred::new();
                return match self.land_alone(v.node, revs, &mut bag) {
                    Ok(landed) => {
                        bag.drain(self);
                        if landed {
                            VPop::Finished(v.node, true)
                        } else {
                            VPop::Next
                        }
                    }
                    Err(payload) => {
                        bag.drain(self);
                        self.vframe_unwound(id, v, v.node, true, payload)
                    }
                };
            }
        } else if !rooted {
            return VPop::Next;
        }
        let mut bag = Deferred::new();
        if v.spec {
            let q = replay::with(|c| {
                let x = c
                    .iter()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                let mut q = v.cidx - x.boundary;
                while q > 0 && x.vstack[q as usize - 1].spec {
                    q -= 1;
                }
                q
            });
            self.drop_chain(id, q, false, &mut bag);
            bag.drain(self);
            return VPop::Restart;
        }
        if !cycle_state::holds(rt, v.node) {
            debug_assert!(
                false,
                "incr-core: a virtual frame roots claims it does not hold"
            );
            return VPop::Next;
        }
        if stand_in.is_some() {
            cycle_state::with_held(|h| {
                if let Some(e) = h.get_mut(&(rt, v.node)) {
                    e.anchor = v.cidx;
                }
            });
        }
        let end = catch_unwind(AssertUnwindSafe(|| {
            self.episode(NodeId(v.node), v.cidx, &mut bag)
        }));
        match end {
            Ok(end) => self.episode_task_end(NodeId(v.node), end, bag),
            Err(payload) => self.vframe_unwound(id, v, v.node, false, payload),
        }
    }

    /// Whether held node `slot`, landing alone in a speculative run, would
    /// land other than Clean, which such a run never does.
    fn alone_lands_dirty(&self, slot: u32) -> bool {
        cycle_state::with_held(|h| {
            let e = &h[&(self.runtime_id, slot)];
            e.poisoned
                || e.last.iter().any(|&(d, _)| {
                    !matches!(
                        state::load::<C>(self.nodes.get(d.0).state_cell()),
                        NodeState::Clean | NodeState::Cyclic
                    )
                })
        })
    }

    /// The end of an episode a virtual frame's pop ran. A dissolved root's
    /// claim lands as if its run never happened, since no frame computes it
    /// here: the entry below does. A suspended episode restarts the driver,
    /// which runs the entries its doom pushed first.
    fn episode_task_end(&self, root: NodeId, end: PopEnd, mut bag: Deferred<C>) -> VPop {
        let pop = match end {
            PopEnd::Suspended => VPop::Restart,
            PopEnd::Landed => VPop::Finished(root.0, true),
            _ => VPop::Next,
        };
        if let PopEnd::Dissolved = end {
            let e = cycle_state::with_held(|h| h.remove(&(self.runtime_id, root.0)));
            let (first, wrote) = e.as_ref().map_or((false, false), |e| (e.first, e.wrote));
            if let Some(e) = e {
                bag.push_erased(Box::new((e.buf, e.s)));
            }
            let stamp_rev = self.current_revision();
            self.discard_land(root, first, wrote, stamp_rev, &mut bag);
        }
        bag.drain(self);
        pop
    }

    /// The frame of `node` at virtual frame `v`'s position ended in an
    /// unwind, out of the run that finished it (`continuation`, after which
    /// the node lands as that frame's compute would) or out of the episode
    /// it rooted, which released its claims. The unwind goes on to the
    /// frame below, a virtual frame's retry or the read that ran the loop.
    fn vframe_unwound(
        &self,
        id: u64,
        v: &VFrame,
        node: u32,
        continuation: bool,
        payload: Box<dyn std::any::Any + Send>,
    ) -> VPop {
        let kind = cycle_state::unwind_kind();
        let mut bag = Deferred::new();
        if continuation {
            self.continuation_unwound(v.node, v.cidx, &mut bag);
        }
        let (pos, below) = replay::with(|c| {
            let x = c
                .iter()
                .find(|x| x.id == id)
                .expect("incr-core: a running context is open");
            let pos = v.cidx - x.boundary;
            (
                pos,
                pos.checked_sub(1).map(|b| x.vstack[b as usize].clone()),
            )
        });
        if v.spec || kind == Some(UnwindKind::Discard) {
            let q = replay::with(|c| {
                let x = c
                    .iter()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                let mut q = pos;
                while q > 0 && x.vstack[q as usize - 1].spec {
                    q -= 1;
                }
                q
            });
            cycle_state::take_unwind_kind();
            bag.push_erased(payload);
            self.drop_chain(id, q, false, &mut bag);
            bag.drain(self);
            return VPop::Restart;
        }
        let Some(below) = below else {
            let boundary =
                replay::with(|c| c.iter().find(|x| x.id == id).map_or(v.cidx, |x| x.boundary));
            // The read that ran the loop reads the target as a compute's
            // unwind leaves its reader.
            self.dep_stack.poison_current_frame(self.runtime_id);
            bag.drain(self);
            self.close_unwound(id, boundary);
            resume_unwind(payload);
        };
        cycle_state::take_unwind_kind();
        replay::with(|c| {
            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                x.deliver_blocked(
                    &below,
                    Delivered::Unwind {
                        cycle: kind == Some(UnwindKind::Cycle),
                        poison: true,
                        payload,
                    },
                );
            }
        });
        bag.drain(self);
        VPop::Finished(node, false)
    }

    /// Lands held node `slot`, whose run at `cidx` ended in an unwind, as
    /// a compute's frame would: the claims anchored at or above `cidx` are
    /// released, and the node lands Failed for a user panic, or takes the
    /// discard landing for an engine unwind or in a speculative run.
    fn continuation_unwound(&self, slot: u32, cidx: u32, bag: &mut Deferred<C>) {
        let rt = self.runtime_id;
        let kind = cycle_state::unwind_kind();
        let discarding = kind.is_some() || replay::speculative();
        let Some(e) = cycle_state::with_held(|h| h.remove(&(rt, slot))) else {
            return;
        };
        self.release_unwound_from(cidx, slot, kind, discarding, bag);
        let id = NodeId(slot);
        let stamp_rev = self.current_revision();
        if discarding {
            self.discard_land(id, e.first, e.wrote, stamp_rev, bag);
        } else {
            let (node, flags) = self.nodes.get_with_flags(slot);
            let attempt = Attempt {
                first: e.first,
                cidx,
                stamp_rev,
                verify_rev: C::walked(&self.walk, &self.revision),
            };
            self.land_failed(id, node, flags, attempt, e.wrote, &e.deps, &[], bag);
        }
        bag.push_erased(Box::new((e.buf, e.s)));
    }

    /// Virtual frame `v` pops: the claims frames above it own keep its
    /// path, so that only its retry takes them over.
    fn note_ancestor(&self, v: &VFrame) {
        cycle_state::with_held(|h| {
            for ((r, _), e) in h.iter_mut() {
                if *r == self.runtime_id && e.owner > v.cidx && e.owner_virtual.is_some() {
                    e.ancestry.push((v.cidx, v.path));
                }
            }
        });
    }

    /// The claims a doomed run of `node` at `cidx` deferred become those of
    /// the run that retries or finishes it there, which releases them if
    /// it unwinds.
    fn own_deferred(&self, cidx: u32, node: u32) {
        cycle_state::with_held(|h| {
            for ((r, _), e) in h.iter_mut() {
                if *r == self.runtime_id
                    && e.owner_virtual.is_some()
                    && e.owner == cidx
                    && e.owner_node == node
                {
                    e.owner_virtual = None;
                }
            }
        });
    }

    /// A run of `node` that finishes its virtual frame's frame starts: the
    /// deliveries held for that frame are read again, in order.
    fn retry_deliveries(&self, id: u64, node: u32, generation: u32, cidx: u32) {
        replay::with(|c| {
            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                for d in x.pending.iter_mut().filter(|d| {
                    d.key.node == node && d.key.generation == generation && d.key.cidx == cidx
                }) {
                    d.taken = false;
                }
            }
        });
    }

    /// That run ended: the deliveries held for its frame are dropped.
    fn end_deliveries(&self, id: u64, node: u32, generation: u32, cidx: u32) {
        let ended: Vec<Delivery> = replay::with(|c| {
            c.iter_mut()
                .find(|x| x.id == id)
                .map(|x| {
                    x.pending
                        .extract_if(.., |d| {
                            d.key.node == node
                                && d.key.generation == generation
                                && d.key.cidx == cidx
                        })
                        .collect()
                })
                .unwrap_or_default()
        });
        drop(ended);
    }

    /// Releases, with the discard landing, the claims virtual hits took
    /// for the entries of context `id` at or above `level` that are still
    /// held.
    fn release_claim_list(&self, id: u64, level: u32, bag: &mut Deferred<C>) {
        let rt = self.runtime_id;
        let slots: Vec<u32> = replay::with(|c| {
            c.iter_mut()
                .find(|x| x.id == id)
                .map(|x| {
                    x.claims
                        .extract_if(.., |&mut (l, _)| l >= level)
                        .map(|(_, slot)| slot)
                        .collect()
                })
                .unwrap_or_default()
        });
        let stamp_rev = self.current_revision();
        for slot in slots {
            if let Some(e) = cycle_state::with_held(|h| h.remove(&(rt, slot))) {
                self.discard_land(NodeId(slot), e.first, e.wrote, stamp_rev, bag);
                bag.push_erased(Box::new((e.buf, e.s)));
            }
        }
    }

    /// An entry's run ended in an unwind. A wound unwinds on. A
    /// speculative entry's panic or cycle is swallowed, and the entry is
    /// dropped (`Some(Gone)`). Any other unwind of an entry above level 0
    /// is held for the virtual frame below it, whose entry's retry raises
    /// it at the read the frame was blocked in (`None`), and the level-0
    /// target's unwinds on to the read that ran the loop.
    fn entry_unwound(
        &self,
        id: u64,
        level: u32,
        entry: Entry,
        boundary: u32,
        payload: Box<dyn std::any::Any + Send>,
    ) -> Option<Computed> {
        let kind = cycle_state::unwind_kind();
        if kind == Some(UnwindKind::Wound) || (!entry.spec && level == 0) {
            self.close_unwound(id, boundary);
            resume_unwind(payload);
        }
        if entry.spec {
            // The speculative chain the entry belongs to starts at the
            // lowest level whose virtual frames below are all real ones'.
            // Dropping only the entry would livelock: the speculative
            // entry below re-reads the node on its retry and pushes it
            // again. The real path's retry reads whatever the unbounded
            // run reads, and raises the panic itself if it gets there.
            let q = replay::with(|c| {
                let x = c
                    .iter()
                    .find(|x| x.id == id)
                    .expect("incr-core: a running context is open");
                let mut q = level;
                while q > 0 && x.vstack[q as usize - 1].spec {
                    q -= 1;
                }
                q
            });
            if q == 0 {
                self.close_unwound(id, boundary);
                resume_unwind(payload);
            }
            cycle_state::take_unwind_kind();
            let mut bag = Deferred::new();
            bag.push_erased(payload);
            self.drop_chain(id, q, false, &mut bag);
            bag.drain(self);
            return None;
        }
        cycle_state::take_unwind_kind();
        replay::with(|c| {
            let x = c
                .iter_mut()
                .find(|x| x.id == id)
                .expect("incr-core: a running context is open");
            let below = x.vstack[level as usize - 1].clone();
            x.deliver_blocked(
                &below,
                Delivered::Unwind {
                    cycle: kind == Some(UnwindKind::Cycle),
                    poison: true,
                    payload,
                },
            );
            x.levels.truncate(level as usize);
            while x.vstack.len() > level as usize {
                x.pop_vframe();
            }
        });
        let mut bag = Deferred::new();
        self.release_held_from(boundary + level, true, &mut bag);
        self.release_claim_list(id, level, &mut bag);
        bag.drain(self);
        None
    }

    /// Drops the chain of context `id` from level `q` up: every entry
    /// above `q`, and the oldest entry at `q`, whose run made the chain,
    /// unless `keep` says it is kept for a retry. The virtual frames from
    /// `q` up are truncated with no Tarjan pops, and the claims held at or
    /// above `q` take the discard landing.
    fn drop_chain(&self, id: u64, q: u32, keep: bool, bag: &mut Deferred<C>) {
        let boundary = replay::with(|c| {
            let x = c
                .iter_mut()
                .find(|x| x.id == id)
                .expect("incr-core: a running context is open");
            x.levels.truncate(q as usize + 1);
            if !keep {
                x.levels[q as usize].pop_front();
            }
            while x.vstack.len() > q as usize {
                x.pop_vframe();
            }
            x.boundary
        });
        self.release_held_from(boundary + q, true, bag);
        self.release_claim_list(id, q, bag);
    }

    /// The unwind held for frame `f`'s read of `handle`, if this is the
    /// read it was held for: a cycle a `try_get` returns, or the unwind
    /// raised again.
    pub(super) fn take_delivery<T: Value>(
        &self,
        f: ActiveRead,
        handle: Incr<T>,
        kind: ReadKind,
    ) -> Option<Result<T, Cycle>> {
        let rt = self.runtime_id;
        let generation = self.nodes.get(f.node.0).generation();
        let reading = u64::from(handle.slot()) | (u64::from(handle.generation()) << 32);
        // A delivery stays until its maker's run ends, because a retry
        // doomed again after the read comes back to it as an entry and must
        // read what the first one did. A payload that can't be copied is
        // delivered once.
        // SAFETY: the active frame's control block is live.
        let member = unsafe { &*f.ctl }.has(cycle::BUFFERED);
        let key = ReadKey {
            node: f.node.0,
            generation,
            cidx: f.cidx,
            reading,
            ordinal: self.dep_stack.reads_of_current(rt),
            path: self.path_at(f.cidx),
            episode: member.then(|| replay::member_run_episode(f.cidx)).flatten(),
        };
        let what = replay::with(|c| {
            c.iter_mut().filter(|x| x.rt == rt).find_map(|x| {
                let at = x.pending.iter().position(|d| {
                    !d.taken
                        && !matches!(d.what, Delivered::Walk(_) | Delivered::Direct)
                        && d.key == key
                })?;
                x.pending[at].taken = true;
                Some(match &x.pending[at].what {
                    Delivered::Landed(value) => {
                        Delivered::Landed(Box::new(value_code::run(|| {
                            value
                                .downcast_ref::<T>()
                                .expect("incr-core: a landed value has its node's type")
                                .clone()
                        })))
                    }
                    Delivered::Walk(_) | Delivered::Direct => {
                        unreachable!("incr-core: a walk's direction is not a value")
                    }
                    Delivered::Unwind {
                        cycle,
                        poison,
                        payload,
                    } => match copy_payload(&**payload) {
                        Some(copy) => Delivered::Unwind {
                            cycle: *cycle,
                            poison: *poison,
                            payload: copy,
                        },
                        None => x.pending.remove(at).what,
                    },
                })
            })
        });
        let Some(what) = what else {
            let slot = handle.slot();
            replay::with(|c| {
                for x in c.iter_mut().filter(|x| x.rt == rt) {
                    x.direct = None;
                    if let Some(d) = x
                        .pending
                        .iter_mut()
                        .find(|d| !d.taken && matches!(d.what, Delivered::Direct) && d.key == key)
                    {
                        d.taken = true;
                        x.direct = Some((slot, handle.generation(), f.cidx));
                    }
                    x.resume = None;
                    if let Some(d) = x
                        .pending
                        .iter_mut()
                        .find(|d| !d.taken && matches!(d.what, Delivered::Walk(_)) && d.key == key)
                    {
                        d.taken = true;
                        if let Delivered::Walk(walk) = &d.what {
                            let walk = (**walk).clone();
                            x.resume = Some((reading, f.cidx, walk));
                        }
                    }
                }
            });
            return None;
        };
        let (cycle, poison, payload) = match what {
            Delivered::Unwind {
                cycle,
                poison,
                payload,
            } => {
                #[cfg(test)]
                self.dirty_reads.lock().unwrap().push(handle.slot());
                (cycle, poison, payload)
            }
            Delivered::Walk(_) | Delivered::Direct => {
                unreachable!("incr-core: a walk's direction is not a value")
            }
            Delivered::Landed(value) => {
                // SAFETY: the active frame's control block is live.
                unsafe { &*f.ctl }.raise(cycle::POISONED);
                return Some(Ok(*value
                    .downcast::<T>()
                    .expect("incr-core: a landed value has its node's type")));
            }
        };
        #[cfg(test)]
        self.deliveries.lock().unwrap().push(DeliverySeen {
            frame: f.node.0,
            read: handle.slot(),
            cycle,
            kind,
        });
        // As at the panicking compute's landing: a closure that catches the
        // unwind, or a `try_get` that returns it, has still read a dep that
        // did not land. The native walk computes a Dirty reader's deps
        // before its closure, so that reader never lands Clean over them.
        if poison {
            // SAFETY: the active frame's control block is live.
            unsafe { &*f.ctl }.raise(cycle::POISONED);
        }
        if !cycle {
            resume_unwind(payload);
        }
        // A closure that swallowed the cycle and then panicked on its own
        // sent that panic as the cycle unwind's payload, which a `try_get`
        // passes on, as `entered` does.
        let payload = match payload.downcast::<crate::cycle::CycleUnwind>() {
            Ok(unwind) if kind == ReadKind::TryGet => return Some(Err(unwind.0)),
            Ok(unwind) => unwind,
            Err(payload) => payload,
        };
        cycle_state::set_unwind_kind(Some(UnwindKind::Cycle));
        resume_unwind(payload)
    }

    /// A virtual frame's pop at position `v.cidx` finished the run of the
    /// frame there, which computed `node`. The frame below reads a landing
    /// as an entry's; if `node` is the entry at the driver's `level`, the
    /// pop was that entry's retry, and the entry is done. `Some` when that
    /// entry was the context's target.
    fn frame_finished(
        &self,
        id: u64,
        level: u32,
        entry: Entry,
        v: &VFrame,
        node: u32,
        landed: bool,
    ) -> Option<Computed> {
        let generation = self.nodes.get(node).generation();
        let (pos, boundary) = replay::with(|c| {
            let x = c
                .iter()
                .find(|x| x.id == id)
                .expect("incr-core: a running context is open");
            (v.cidx - x.boundary, x.boundary)
        });
        if landed && pos > 0 {
            self.hold_landing(id, pos, node, generation);
        }
        if pos != level || entry.node != node || entry.generation != generation {
            return None;
        }
        replay::with(|c| {
            if let Some(x) = c.iter_mut().find(|x| x.id == id) {
                x.pop_entry(level);
                x.claims.retain(|&(l, _)| l < level);
            }
        });
        if level == 0 {
            self.close_context(id, boundary);
            // The read that ran the loop reads the target as the native
            // compute's reader does after its landing.
            if landed {
                self.episode_landed(self.nodes.get(node));
            }
            return Some(Computed::Done);
        }
        None
    }

    /// Node `node` at `generation` landed from position `level`. If it
    /// landed Dirty and the frame below was blocked reading it, that
    /// frame's retry reads the slot as it is, as the unbounded run reads
    /// it right after the compute. If an episode landed it with no value,
    /// that read gets the episode's cycle, as the read that ran it would.
    fn hold_landing(&self, id: u64, level: u32, node: u32, generation: u32) {
        let state = state::load::<C>(self.nodes.get(node).state_cell());
        let reading = u64::from(node) | (u64::from(generation) << 32);
        let what = match state {
            NodeState::Dirty => self.slot_value(node).map(Delivered::Landed),
            NodeState::New | NodeState::Cyclic => {
                self.inner
                    .read()
                    .member_recs
                    .get(&node)
                    .map(|g| Delivered::Unwind {
                        cycle: true,
                        // As `episode_landed` poisons a reader of a node
                        // left New, and not of one left Cyclic.
                        poison: state == NodeState::New,
                        payload: Box::new(cycle::CycleUnwind(g.cycle.clone())),
                    })
            }
            _ => None,
        };
        replay::with(|c| {
            let Some(x) = c.iter_mut().find(|x| x.id == id) else {
                return;
            };
            let below = x.vstack[level as usize - 1].clone();
            // A walk the doom stopped short of its root goes on in the
            // retry after this compute, as the unbounded run's goes on.
            let key = ReadKey::blocked(&below);
            let walk = x.pending.iter_mut().find_map(|d| match &mut d.what {
                Delivered::Walk(walk) if d.key == key => Some(walk),
                _ => None,
            });
            match (walk, what) {
                (Some(walk), _) => walk.ran(node),
                (None, Some(what)) if below.reading == reading => x.deliver_blocked(&below, what),
                _ => {}
            }
        });
    }

    /// Closes context `id` on an unwind out of its loop, which also ends
    /// the runs whose virtual hits took the claims it lists.
    fn close_unwound(&self, id: u64, boundary: u32) {
        let mut bag = Deferred::new();
        self.release_claim_list(id, 0, &mut bag);
        bag.drain(self);
        self.close_context(id, boundary);
    }

    fn close_context(&self, id: u64, boundary: u32) {
        if value_code::thread_word() & value_code::MEMO_BIT != 0 {
            taint_memo::clear_from(boundary);
        }
        replay::close(id);
    }

    /// [`Self::ensure_clean`] for a driver entry, whose walk skips nodes
    /// with virtual frames.
    fn ensure_clean_virt(&self, id: NodeId, gen: u32) -> Computed {
        let word = value_code::checked_word();
        let observed = state::load::<C>(self.nodes.get(id.0).state_cell());
        if observed == NodeState::Clean {
            return Computed::Done;
        }
        self.clean_walk::<VIRT>(id, gen, word, observed)
    }

    /// Whether a `Direct` delivery sends this read's walk of `id` straight
    /// to its compute.
    #[cold]
    #[inline(never)]
    pub(super) fn take_direct(&self, id: NodeId) -> bool {
        let rt = self.runtime_id;
        let Some((_, at)) = self.dep_stack.active_frame(rt) else {
            return false;
        };
        let key = (id.0, self.nodes.get(id.0).generation(), at);
        replay::with(|c| {
            c.iter_mut()
                .filter(|x| x.rt == rt)
                .any(|x| x.direct.take_if(|d| *d == key).is_some())
        })
    }

    /// The path of the frame at `cidx` if a virtual frame at or above it
    /// owns claims, which only a run along the same path takes over.
    pub(super) fn virtual_path_at(&self, cidx: u32) -> Option<u64> {
        if value_code::thread_word() & value_code::REPLAY_BIT == 0 {
            return None;
        }
        let rt = self.runtime_id;
        cycle_state::with_held(|h| {
            h.iter()
                .any(|((r, _), e)| *r == rt && e.owner >= cidx && e.owner_virtual.is_some())
        })
        .then(|| self.path_at(cidx))
    }

    /// The path of a frame at position `cidx`: the highest virtual frame
    /// below it extended by the frames on the stack between them.
    fn path_at(&self, cidx: u32) -> u64 {
        let rt = self.runtime_id;
        let base = replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .flat_map(|x| x.vstack.iter())
                .filter(|v| v.cidx < cidx)
                .max_by_key(|v| v.cidx)
                .map(|v| (v.cidx, replay::extend_path(v.path, (v.reading, v.ordinal))))
        });
        let (from, path) = base.map_or((0, 0), |(at, path)| (at + 1, path));
        self.dep_stack
            .read_steps(rt, from, cidx)
            .into_iter()
            .fold(path, replay::extend_path)
    }

    /// The lowlink a popped virtual frame of `node` at `cidx`, along this
    /// frame's path, left for the run that finishes it.
    #[cold]
    #[inline(never)]
    pub(super) fn take_low(&self, node: NodeId, cidx: u32) -> Option<u32> {
        let rt = self.runtime_id;
        let generation = self.nodes.get(node.0).generation();
        let key = |l: &(u32, u32, u32, u64, u32)| l.0 == node.0 && l.1 == generation && l.2 == cidx;
        if !replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .any(|x| x.lows.iter().any(key))
        }) {
            return None;
        }
        let path = self.path_at(cidx);
        replay::with(|c| {
            c.iter_mut().filter(|x| x.rt == rt).find_map(|x| {
                let at = x.lows.iter().position(|l| key(l) && l.3 == path)?;
                Some(x.lows.remove(at).4)
            })
        })
    }

    /// The walk this read of `root` resumes: a frame's retry making the
    /// read whose walk a doom stopped goes on from where it stopped.
    #[cold]
    #[inline(never)]
    pub(super) fn take_resume(&self, root: NodeId) -> Option<Walk> {
        let rt = self.runtime_id;
        let (_, at) = self.dep_stack.active_frame(rt)?;
        let reading = u64::from(root.0) | (u64::from(self.nodes.get(root.0).generation()) << 32);
        replay::with(|c| {
            c.iter_mut()
                .filter(|x| x.rt == rt)
                .find_map(|x| x.resume.take_if(|r| r.0 == reading && r.1 == at))
                .map(|r| r.2)
        })
    }

    /// A walk of `root` whose compute of `entry` came back other than
    /// `Done`: `Ok` with the result the walk goes on with, or `Err` with
    /// the one it stops on, its rest stashed for the reading frame's retry.
    /// Takes the walk by reference so the hot loop never moves it.
    #[cold]
    #[inline(never)]
    pub(super) fn walk_stopped(
        &self,
        root: NodeId,
        entry: super::Entry,
        visit: Visit,
        is_root: bool,
        walk: &mut Walk,
        computed: Computed,
    ) -> Result<Computed, Computed> {
        match computed {
            Computed::Doomed => {
                self.stash_walk(root, (!is_root).then_some(walk));
                Err(computed)
            }
            Computed::Discarded => match self.walk_discarded(visit) {
                Some(after) => Ok(after),
                None => {
                    // The discarded compute is a frame the reading frame's
                    // retry runs again.
                    walk.push(entry);
                    self.stash_walk(root, (!is_root).then_some(walk));
                    Err(computed)
                }
            },
            _ => Ok(computed),
        }
    }

    /// A doom stopped the reading frame's walk of `root`: its retry
    /// resumes `walk`, the rest of it, or with none left, takes the root's
    /// landing as the read's outcome.
    fn stash_walk(&self, root: NodeId, walk: Option<&mut Walk>) {
        // A doomed frame's walk reads past placeholders, and a speculative
        // run's retry takes no delivery.
        if replay::speculative() {
            return;
        }
        let rt = self.runtime_id;
        let Some((_, at)) = self.dep_stack.active_frame(rt) else {
            return;
        };
        let reading = u64::from(root.0) | (u64::from(self.nodes.get(root.0).generation()) << 32);
        replay::with(|c| {
            for x in c.iter_mut().filter(|x| x.rt == rt).rev() {
                if let Some(v) = x
                    .vstack
                    .iter()
                    .rev()
                    .find(|v| v.cidx == at && v.reading == reading)
                    .cloned()
                {
                    match walk {
                        Some(walk) => x.deliver_walk(&v, std::mem::take(walk)),
                        None => {
                            let key = ReadKey::blocked(&v);
                            x.pending.retain(|d| {
                                !(matches!(d.what, Delivered::Walk(_)) && d.key == key)
                            });
                        }
                    }
                    return;
                }
            }
        });
    }

    /// The taint-memo owner for a walk's skip of `id`, if it has a virtual
    /// frame in a context of this runtime: one more than that frame's
    /// position, as for a skip of a frame on the stack, so that its pop
    /// clears what the skip memoized.
    #[cold]
    #[inline(never)]
    pub(super) fn virtual_owner(&self, id: NodeId) -> Option<u32> {
        self.virtual_position(id.0, self.nodes.get(id.0).generation())
            .map(|p| p + 1)
    }

    /// Whether position `cidx` holds one of this runtime's virtual frames.
    #[cfg(all(test, not(loom)))]
    pub(super) fn is_virtual_cidx(&self, cidx: u32) -> bool {
        let rt = self.runtime_id;
        replay::with(|c| {
            c.iter()
                .filter(|x| x.rt == rt)
                .any(|x| x.vstack.iter().any(|v| v.cidx == cidx))
        })
    }
}

/// A copy of a panic payload of a type the engine knows: a cycle unwind's
/// or a `panic!` message.
fn copy_payload(payload: &(dyn std::any::Any + Send)) -> Option<Box<dyn std::any::Any + Send>> {
    if let Some(unwind) = payload.downcast_ref::<crate::cycle::CycleUnwind>() {
        return Some(Box::new(crate::cycle::CycleUnwind(unwind.0.clone())));
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return Some(Box::new(message.clone()));
    }
    payload
        .downcast_ref::<&'static str>()
        .map(|message| Box::new(*message) as Box<dyn std::any::Any + Send>)
}
