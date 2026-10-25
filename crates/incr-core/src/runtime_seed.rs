//! Restore and checkpoint hooks for incr-persist: `seed_query`, which
//! lands an unpublished query Dirty with a stored value and deps, and
//! `inspect`, which reads a Clean query's value with the deps of the run
//! that produced it. Neither computes, and `seed_query` claims only a
//! node no reader has claimed.

use super::*;
use crate::persist_hooks::{Inspected, Reuse, SeedBase, SeedDep, SeedError};

impl<C: Cells> Runtime<C> {
    pub(crate) fn seed_base(&self) -> SeedBase {
        SeedBase {
            w0: NodeData::<C>::input_stamp(C::walked(&self.walk, &self.revision)),
            rid: self.runtime_id,
        }
    }

    pub(crate) fn seed_query<T: Value>(
        &self,
        base: &SeedBase,
        q: Incr<T>,
        value: T,
        deps: &[SeedDep],
        reuse: Reuse,
    ) -> Result<Reuse, (SeedError, T)> {
        value_code::check();
        if base.rid != self.runtime_id || q.runtime_id() != self.runtime_id {
            return Err((SeedError::ForeignBase, value));
        }
        assert!(
            !self.frame_doomed(),
            "incr-core: seed_query called in a doomed frame"
        );
        let id = NodeId(q.slot());
        let node = self.nodes.get(id.0);
        if state::try_transition::<C>(node.state_cell(), NodeState::New, NodeState::Computing)
            .is_err()
        {
            return Err((SeedError::NotFresh, value));
        }
        // Another thread's doom keeps the node as its unbounded run's frame
        // does, claimed.
        if self.reserved_elsewhere(id, q.generation()) {
            self.unclaim_unrun(node, false);
            return Err((SeedError::NotFresh, value));
        }
        let installed = {
            let mut inner = self.inner.write();
            self.install_seed(&mut inner, id, q.generation(), deps, reuse)
        };
        let reuse = match installed {
            Ok(reuse) => reuse,
            Err(err) => {
                self.land(id, node, NodeState::New, NodeState::New);
                return Err((err, value));
            }
        };
        let arena = self
            .arenas
            .tagged_arena::<T>(node.type_tag())
            .expect("incr-core: a query's arena exists once it is created");
        drop(arena.write(node.arena_slot(), value));
        node.set_changed_at(base.w0);
        node.set_verified_at(match reuse {
            Reuse::IfUnchanged => base.w0,
            Reuse::RecomputeFirst => 0,
        });
        self.land(id, node, NodeState::Dirty, NodeState::Dirty);
        Ok(reuse)
    }

    /// `seed_query`'s checks and edges under its `inner.write()`, with
    /// `q` claimed: the `Reuse` it lands with, or why it can't seed.
    fn install_seed(
        &self,
        inner: &mut Inner<C>,
        id: NodeId,
        generation: u32,
        deps: &[SeedDep],
        reuse: Reuse,
    ) -> Result<Reuse, SeedError> {
        let node = self.nodes.get(id.0);
        if node.generation() != generation
            || node.dep_count() != 0
            || !inner.dependents[id.0 as usize].is_empty()
        {
            return Err(SeedError::NotFresh);
        }
        // The runtime first: a foreign dep's slot may lie past this
        // runtime's nodes.
        let alive = deps.iter().all(|d| {
            d.dep.runtime_id() == self.runtime_id && {
                let dep = self.nodes.get(d.dep.slot());
                dep.generation() == d.dep.generation() && dep.state() != NodeState::Dead
            }
        });
        if !alive {
            return Err(SeedError::DepNotAlive);
        }
        if deps.is_empty() && reuse == Reuse::RecomputeFirst {
            return Err(SeedError::NoDeps);
        }
        let reuse = match reuse {
            Reuse::IfUnchanged if !deps.iter().all(|d| self.trusts(d)) => Reuse::RecomputeFirst,
            reuse => reuse,
        };
        // `RecomputeFirst` lands with `verified_at` 0, which always runs, so
        // its recorded deps would only make a walk compute ones the rerun
        // no longer reads, which a cold run never does.
        if reuse == Reuse::IfUnchanged {
            let ids: Vec<NodeId> = deps.iter().map(|d| NodeId(d.dep.slot())).collect();
            let displaced = node.install_deps(&ids);
            debug_assert!(displaced.is_none(), "a fresh query has no dep list");
            for d in &ids {
                inner.dependents[d.0 as usize].push_unique(id);
            }
        }
        Ok(reuse)
    }

    /// Whether a dep holds what the seed recorded: a mapped one always
    /// does (its fingerprint was checked by the caller); a query dep only
    /// if it is Clean or Dirty at its seed stamp. The check is an RMW, so
    /// a later claim of the dep reads from it and stamps above W0.
    fn trusts(&self, d: &SeedDep) -> bool {
        let Some(stamp) = d.trusted_stamp else {
            return true;
        };
        let id = NodeId(d.dep.slot());
        let dep = self.nodes.get(id.0);
        self.park(ParkPoint::SeedBeforeTrust(id));
        let cur = state::load::<C>(dep.state_cell());
        let held = matches!(cur, NodeState::Clean | NodeState::Dirty)
            && state::try_transition::<C>(dep.state_cell(), cur, cur).is_ok()
            && dep.changed_at() == stamp;
        self.park(ParkPoint::SeedAfterTrust(id));
        held
    }

    pub(crate) fn inspect<T: Value>(&self, q: Incr<T>) -> Option<Inspected<T>> {
        assert!(
            !self.dep_stack.current_frame_active(self.runtime_id),
            "incr-core: inspect called during compute"
        );
        if q.runtime_id() != self.runtime_id {
            return None;
        }
        let id = NodeId(q.slot());
        let (node, flags) = self.nodes.get_with_flags(id.0);
        let first = {
            let _inner = self.inner.read();
            read_under(node, q.generation())
        };
        let (stamp, deps) = first?;
        self.park(ParkPoint::InspectBetweenHolds(id));
        let type_tag = node.type_tag();
        let arena_slot = node.arena_slot();
        C::fence_acquire();
        if node.generation() != q.generation() {
            return None;
        }
        let arena = self.arenas.tagged_arena::<T>(type_tag);
        let value = value_code::run(|| arena.and_then(|arena| arena.try_read(arena_slot)))?;
        // The value drops only after this hold if a check fails.
        let checked = {
            let _inner = self.inner.read();
            read_under(node, q.generation())
                .filter(|again| *again == (stamp, deps))
                .map(|(_, deps)| {
                    let deps = deps
                        .iter()
                        .map(|d| {
                            ErasedIncr::new(d.0, self.nodes.get(d.0).generation(), self.runtime_id)
                        })
                        .collect();
                    (deps, C::u8_load_relaxed(flags) & IN_GROUP != 0)
                })
        };
        let (deps, in_group) = checked?;
        Some(Inspected {
            value,
            changed_at: stamp,
            deps,
            in_group,
        })
    }
}

/// `inspect`'s reads under a hold its caller has taken.
fn read_under<C: Cells>(node: &NodeData<C>, generation: u32) -> Option<(u64, Vec<NodeId>)> {
    if node.generation() != generation || state::load::<C>(node.state_cell()) != NodeState::Clean {
        return None;
    }
    let mut deps = Vec::with_capacity(node.dep_count() as usize);
    node.for_each_dep(|d| deps.push(d));
    Some((node.changed_at(), deps))
}
