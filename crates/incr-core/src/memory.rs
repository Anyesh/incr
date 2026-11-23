//! Memory figures for a runtime, computed by [`Runtime::memory_stats`]
//! when it is called. Nothing here runs on a read, a set, propagation or
//! a collection change: state registers once, when its collection or
//! operator is created, and the walk visits nodes and registered state
//! only on request.
//!
//! Byte figures are estimates from sizes and capacities. A node's bytes
//! are its slot in the node store, its flags byte, its compute-function
//! and dependents entries, its value slot (and the box a `Shared` slot
//! keeps its value in), its compute closure's captures, its overflow dep
//! list, its dependents list's heap block and index, and its label. A
//! state's bytes are its allocation and the heap tables, vectors and logs
//! it owns, sized by capacity. Neither counts heap memory a value owns
//! behind its own pointers (a `String`'s text, a Python object), memory
//! the allocator keeps beyond what was asked for, or per-runtime tables
//! whose size does not grow with nodes.

use std::any::TypeId;
use std::sync::{Arc, Weak};

use crate::cells::Cells;
use crate::collection::OpLock;
use crate::dep_stack::DepStack;
use crate::handle::ErasedIncr;
use crate::locks::Lock;
use crate::node::{NodeData, NodeId};
use crate::runtime::{ComputeFn, Runtime};
use crate::state::{self, NodeState};
use crate::value_code;

/// What a node is, for [`NodeMemory`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    /// A node set from outside, with no compute function.
    Input,
    /// A node with a compute function.
    Query,
    /// The version node of a collection (a source, an operator output or a
    /// `group_by` group), whose state is listed in [`MemoryStats::states`].
    Collection,
}

/// One live node. A [`MemoryVisitor`] sees it with the label borrowed
/// (`NodeMemory<&str>`); [`MemoryStats`] keeps it owned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeMemory<L = String> {
    pub slot: u32,
    pub kind: NodeKind,
    /// Its value type, looked up with [`MemoryStats::value_type`].
    pub type_tag: u16,
    pub has_compute: bool,
    /// Whether its value slot holds a value (a query that never ran has
    /// none).
    pub has_value: bool,
    pub label: Option<L>,
    pub bytes: usize,
}

impl NodeMemory<&str> {
    pub fn into_owned(self) -> NodeMemory {
        NodeMemory {
            slot: self.slot,
            kind: self.kind,
            type_tag: self.type_tag,
            has_compute: self.has_compute,
            has_value: self.has_value,
            label: self.label.map(str::to_owned),
            bytes: self.bytes,
        }
    }
}

/// A value type that has an arena in the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueType {
    pub type_id: TypeId,
    pub name: &'static str,
    /// The bytes of one value slot.
    pub slot_bytes: usize,
}

/// What a piece of registered state belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StateKind {
    /// A collection's log: its live elements and pending deltas.
    Collection,
    /// `group_by`'s map from key to group.
    GroupBy,
    /// The key each element was filed under, kept by `group_by`.
    KeyCache,
    /// `aggregate`'s tree and element slots.
    Aggregate,
    /// Both sides of a `join`, by key.
    Join,
    /// `sort_by_key`'s sorted elements and positional log.
    Sort,
    /// The copy of the sorted order a `pairwise` or `window` keeps.
    SortMirror,
    /// State registered from outside incr-core.
    Other(&'static str),
}

/// How many values of one type a piece of state holds, so a caller can
/// count values of a type it knows (by `TypeId`) without incr-core
/// knowing that type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Held {
    pub type_id: TypeId,
    pub count: usize,
}

impl Held {
    pub fn of<T: 'static>(count: usize) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            count,
        }
    }
}

/// One piece of registered state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateMemory {
    pub kind: StateKind,
    /// The live node it was registered with, filled in by the walk.
    pub node: Option<NodeId>,
    /// Live elements (a collection's, with multiset copies counted), or
    /// entries for a map.
    pub elements: usize,
    /// Deltas logged and not yet compacted away.
    pub pending_deltas: usize,
    /// The values it holds, by type: a distinct element once however
    /// many copies it has, each logged delta once.
    pub held: Vec<Held>,
    pub bytes: usize,
}

impl StateMemory {
    pub fn new(kind: StateKind, elements: usize, held: Vec<Held>, bytes: usize) -> Self {
        Self {
            kind,
            node: None,
            elements,
            pending_deltas: 0,
            held,
            bytes,
        }
    }

    /// How many values of `type_id` it holds.
    pub fn held(&self, type_id: TypeId) -> usize {
        self.held
            .iter()
            .filter(|h| h.type_id == type_id)
            .map(|h| h.count)
            .sum()
    }
}

/// State that reports its own figures to [`Runtime::memory_stats`],
/// registered with [`Runtime::register_state`]. Called with no incr lock
/// held; it must not call into the runtime.
pub trait StateStats {
    fn state_stats(&self) -> StateMemory;
}

/// What [`Runtime::memory_stats`] found. See the module docs for what the
/// byte figures include. A [`MemoryVisitor`] that keeps everything, so
/// `MemoryStats::default()` stepped by a [`MemoryWalker`] to its end
/// holds what `memory_stats` returns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryStats {
    /// Every live node, by slot.
    pub nodes: Vec<NodeMemory>,
    /// Every registered piece of state still alive.
    pub states: Vec<StateMemory>,
    /// Value types, indexed by tag.
    pub value_types: Vec<ValueType>,
    /// Slots of deleted nodes waiting to be reused.
    pub free_slots: usize,
    /// The bytes those slots keep.
    pub free_bytes: usize,
    /// The traces kept for `get_traced` roots, which [`Self::bytes`]
    /// leaves out.
    pub kept_traces: crate::kept_trace::KeptTraceMemory,
}

impl MemoryStats {
    pub fn value_type(&self, tag: u16) -> Option<&ValueType> {
        self.value_types.get(tag as usize)
    }

    pub fn node_bytes(&self) -> usize {
        self.nodes.iter().map(|n| n.bytes).sum()
    }

    pub fn state_bytes(&self) -> usize {
        self.states.iter().map(|s| s.bytes).sum()
    }

    /// Every byte figure: nodes, state and free slots.
    pub fn bytes(&self) -> usize {
        self.node_bytes() + self.state_bytes() + self.free_bytes
    }

    /// Whether `node`'s value is a `type_id`, and it holds one.
    pub fn holds_value_of(&self, node: &NodeMemory, type_id: TypeId) -> bool {
        node.has_value
            && self
                .value_type(node.type_tag)
                .is_some_and(|t| t.type_id == type_id)
    }

    /// How many values of `type_id` nodes and state hold.
    pub fn held(&self, type_id: TypeId) -> usize {
        let in_nodes = self
            .nodes
            .iter()
            .filter(|n| self.holds_value_of(n, type_id))
            .count();
        in_nodes + self.states.iter().map(|s| s.held(type_id)).sum::<usize>()
    }
}

impl MemoryVisitor for MemoryStats {
    fn state(&mut self, state: StateMemory, _labels: &Labels<'_>) {
        self.states.push(state);
    }

    fn node(&mut self, node: NodeMemory<&str>, _value_type: Option<&ValueType>) {
        self.nodes.push(node.into_owned());
    }

    fn finish(&mut self, end: WalkEnd) {
        self.value_types = end.value_types;
        self.free_slots = end.free_slots;
        self.free_bytes = end.free_bytes;
        self.kept_traces = end.kept_traces;
    }
}

type Registered = (Weak<dyn StateStats + Send + Sync>, Option<ErasedIncr>);

/// The runtime's registered state. Dead entries are dropped by a walk,
/// and by a registration once the list has doubled since the last prune,
/// so a runtime that is never walked keeps at most about twice its live
/// entries.
pub(crate) struct StateRegistry {
    entries: Vec<Registered>,
    prune_at: usize,
}

const FIRST_PRUNE: usize = 64;

impl StateRegistry {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            prune_at: FIRST_PRUNE,
        }
    }

    pub(crate) fn register(
        &mut self,
        node: Option<ErasedIncr>,
        state: Weak<dyn StateStats + Send + Sync>,
    ) {
        if self.entries.len() >= self.prune_at {
            self.prune();
            self.prune_at = (2 * self.entries.len()).max(FIRST_PRUNE);
        }
        self.entries.push((state, node));
    }

    fn prune(&mut self) {
        self.entries.retain(|(state, _)| state.strong_count() > 0);
    }

    /// The live entries, still weak, after dropping the dead ones.
    pub(crate) fn snapshot(&mut self) -> Vec<Registered> {
        self.prune();
        self.entries.clone()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<C: Cells> Runtime<C> {
    /// Adds `state` to what [`Self::memory_stats`] reports, against `node`
    /// while that node lives. The runtime keeps only the `Weak`: state
    /// dropped elsewhere leaves the report.
    pub fn register_state(
        &self,
        node: Option<ErasedIncr>,
        state: Weak<dyn StateStats + Send + Sync>,
    ) {
        self.states.write().register(node, state);
    }

    /// Walks every live node and every registered piece of state still
    /// alive: one [`MemoryWalker`] taken to its end. Costs time in the
    /// number of node slots and state entries; nothing is kept between
    /// calls. Not allowed from value code, or from an `aggregate`'s
    /// `combine`, which runs under that aggregate's lock.
    pub fn memory_stats(&self) -> MemoryStats {
        let mut stats = MemoryStats::default();
        self.memory_walker().step(self, &mut stats, None);
        stats
    }

    /// Registry entries, dead ones included.
    #[cfg(test)]
    pub(crate) fn registered_states(&self) -> usize {
        self.states.read().len()
    }

    /// A walk for [`MemoryWalker::step`] to take in parts.
    pub fn memory_walker(&self) -> MemoryWalker {
        MemoryWalker::new(self)
    }
}

/// The labels of live nodes, for a [`MemoryVisitor`] naming the node a
/// state belongs to.
pub struct Labels<'a>(&'a std::collections::HashMap<u32, String>);

impl<'a> Labels<'a> {
    pub fn get(&self, slot: u32) -> Option<&'a str> {
        self.0.get(&slot).map(String::as_str)
    }
}

/// What a walk reads once, when it ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalkEnd {
    pub value_types: Vec<ValueType>,
    pub free_slots: usize,
    pub free_bytes: usize,
    pub kept_traces: crate::kept_trace::KeptTraceMemory,
}

/// Receives what a [`MemoryWalker`] measures: every registered state
/// still alive, then every live node, then [`WalkEnd`]. `state` and
/// `node` run under the runtime's read lock and must not call into the
/// runtime.
pub trait MemoryVisitor {
    fn state(&mut self, state: StateMemory, labels: &Labels<'_>);
    fn node(&mut self, node: NodeMemory<&str>, value_type: Option<&ValueType>);
    fn finish(&mut self, end: WalkEnd);
}

/// [`Runtime::memory_stats`] in parts: each [`Self::step`] measures
/// states and slots until its deadline passes, so the caller gets its
/// thread back between steps. The states are those registered when the
/// walk began, measured if still alive when reached; the slots are those
/// that existed then, each read as it is when reached. On a runtime that
/// does not change during the walk, the figures equal one
/// `memory_stats`; otherwise they can mix moments.
pub struct MemoryWalker {
    runtime_id: crate::handle::RuntimeId,
    states: Vec<Registered>,
    next_state: usize,
    /// Slots below this existed when the walk began.
    len: u32,
    next_slot: u32,
    batch: u32,
    collection_slot: Vec<bool>,
    value_types: Vec<ValueType>,
    fixed: usize,
    free_slots: usize,
    finished: bool,
}

/// Slots read under one hold of the runtime's read lock, between clock
/// checks.
const SLOT_BATCH: u32 = 4096;

impl MemoryWalker {
    fn new<C: Cells>(rt: &Runtime<C>) -> Self {
        value_code::check();
        let states = rt.states.write().snapshot();
        // Read after the snapshot: a registered node's slot predates its registration.
        let len = rt.nodes.len();
        Self {
            runtime_id: rt.runtime_id,
            states,
            next_state: 0,
            len,
            next_slot: 0,
            batch: SLOT_BATCH,
            collection_slot: vec![false; len as usize],
            value_types: Vec::new(),
            fixed: std::mem::size_of::<NodeData<C>>()
                + 1
                + std::mem::size_of::<Option<ComputeFn<C>>>()
                + std::mem::size_of::<crate::dependent_list::DependentList>(),
            free_slots: 0,
            finished: false,
        }
    }

    /// Whether `rt` is the runtime this walk measures.
    pub fn belongs_to<C: Cells>(&self, rt: &Runtime<C>) -> bool {
        rt.runtime_id == self.runtime_id
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Measures until `deadline` passes (at least one state or one batch
    /// of slots), or to the end with `None`, and returns whether the walk
    /// has ended; `visitor` has then had [`MemoryVisitor::finish`]. Panics
    /// if `rt` is not the runtime the walk began on. Not allowed where
    /// [`Runtime::memory_stats`] is not.
    pub fn step<C: Cells>(
        &mut self,
        rt: &Runtime<C>,
        visitor: &mut impl MemoryVisitor,
        deadline: Option<std::time::Instant>,
    ) -> bool {
        assert!(
            self.belongs_to(rt),
            "a MemoryWalker steps only the runtime that made it"
        );
        value_code::check();
        rt.dep_stack.unrecorded_lookup();
        while !self.finished {
            if self.next_state < self.states.len() {
                self.visit_state(rt, visitor);
            } else if self.next_slot < self.len {
                self.visit_slots(rt, visitor);
            } else {
                self.finish(rt, visitor);
                break;
            }
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                break;
            }
        }
        self.finished
    }

    fn visit_state<C: Cells>(&mut self, rt: &Runtime<C>, visitor: &mut impl MemoryVisitor) {
        let (state, node) = &self.states[self.next_state];
        self.next_state += 1;
        let Some(state) = state.upgrade() else {
            return;
        };
        let mut stats = state.state_stats();
        drop(state);
        stats.node = node.filter(|&n| rt.alive(n)).map(|n| NodeId(n.slot()));
        if let (StateKind::Collection, Some(n)) = (stats.kind, stats.node) {
            if let Some(seen) = self.collection_slot.get_mut(n.0 as usize) {
                *seen = true;
            }
        }
        let inner = rt.inner.read();
        visitor.state(stats, &Labels(&inner.labels));
    }

    fn visit_slots<C: Cells>(&mut self, rt: &Runtime<C>, visitor: &mut impl MemoryVisitor) {
        let end = self.len.min(self.next_slot.saturating_add(self.batch));
        let inner = rt.inner.read();
        for slot in self.next_slot..end {
            let node = rt.nodes.get(slot);
            if state::load::<C>(node.state_cell()) == NodeState::Dead {
                self.free_slots += 1;
                continue;
            }
            let tag = node.type_tag();
            if tag as usize >= self.value_types.len() {
                self.value_types = rt.arenas.value_types();
            }
            let value_type = self.value_types.get(tag as usize);
            let filled = rt.arenas.erased(tag).filled_bytes(node.arena_slot());
            let compute = inner
                .compute_fns
                .get(slot as usize)
                .and_then(|f| f.as_ref());
            let label = inner.labels.get(&slot);
            let bytes = self.fixed
                + value_type.map_or(0, |t| t.slot_bytes)
                + filled.unwrap_or(0)
                + compute.map_or(0, |f| std::mem::size_of_val(&**f))
                + node.overflow_dep_bytes()
                + inner
                    .dependents
                    .get(slot as usize)
                    .map_or(0, |d| d.heap_bytes())
                + label.map_or(0, |l| {
                    l.capacity() + std::mem::size_of::<(u32, String)>() + 1
                });
            let kind = if self.collection_slot[slot as usize] {
                NodeKind::Collection
            } else if compute.is_some() {
                NodeKind::Query
            } else {
                NodeKind::Input
            };
            visitor.node(
                NodeMemory {
                    slot,
                    kind,
                    type_tag: tag,
                    has_compute: compute.is_some(),
                    has_value: filled.is_some(),
                    label: label.map(String::as_str),
                    bytes,
                },
                value_type,
            );
        }
        self.next_slot = end;
    }

    fn finish<C: Cells>(&mut self, rt: &Runtime<C>, visitor: &mut impl MemoryVisitor) {
        self.finished = true;
        self.states = Vec::new();
        self.collection_slot = Vec::new();
        let kept_traces = rt.inner.read().last_traces.memory();
        visitor.finish(WalkEnd {
            value_types: rt.arenas.value_types(),
            free_slots: self.free_slots,
            free_bytes: self.free_slots * self.fixed,
            kept_traces,
        });
    }

    #[cfg(test)]
    pub(crate) fn set_batch(&mut self, slots: u32) {
        self.batch = slots;
    }
}

/// The figures of state kept behind an [`OpLock`]: `bytes` counts only
/// the heap it owns, as the lock and its `Arc` are added by the caller.
pub(crate) trait Measure {
    fn measure(&self) -> StateMemory;
}

impl<M: Measure + 'static, C: Cells> StateStats for OpLock<M, C> {
    fn state_stats(&self) -> StateMemory {
        let mut stats = self.read().measure();
        stats.bytes += std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>();
        stats
    }
}

/// Registers `state` with `rt`, against `node`.
pub(crate) fn register<M: Measure + 'static, C: Cells>(
    rt: &Runtime<C>,
    node: ErasedIncr,
    state: &Arc<OpLock<M, C>>,
) {
    let weak: Weak<OpLock<M, C>> = Arc::downgrade(state);
    rt.register_state(Some(node), weak);
}

/// The bytes of a hash table of `capacity` entries of `E`: its buckets
/// and one control byte each, as hashbrown lays them out.
pub(crate) fn table_bytes<E>(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    let buckets = (capacity * 8 / 7).next_power_of_two();
    buckets * (std::mem::size_of::<E>() + 1)
}

/// The bytes of a `Vec<E>`'s buffer.
pub(crate) fn vec_bytes<E>(v: &Vec<E>) -> usize {
    v.capacity() * std::mem::size_of::<E>()
}
