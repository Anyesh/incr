//! A kept `get_traced` trace in one allocation, behind a thin pointer.
//!
//! The block holds the [`Header`], then per step its `elapsed_ns`
//! (`u64`), its slot, its slot's generation and the end of its
//! dependencies (`u32` each; the end's top four bits hold the step's
//! action and value kind), then every step's dependency slots (`u32`),
//! then the value words (`u64`): one for a count or a reference, two for
//! a pair, none for no value.

use std::alloc::{self, Layout};
use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{self, AtomicU32, AtomicU64, Ordering};

use crate::node::NodeId;
use crate::trace::{NodeTrace, PropagationTrace, StampPair, TraceAction};

/// A step's value as a binding keeps it: none, a native `u64` or
/// `(u64, u64)`, or a reference of the binding's own type, at most eight
/// bytes and eight-aligned.
pub enum StepValue<V> {
    None,
    Count(u64),
    Pair(u64, u64),
    Ref(V),
}

const END_BITS: u32 = 28;
const END_MASK: u32 = (1 << END_BITS) - 1;
const VERIFIED: u32 = 0;
const CHANGED: u32 = 1;
const CUTOFF: u32 = 2;
const NO_VALUE: u32 = 0;
const COUNT: u32 = 1;
const PAIR: u32 = 2;
const REF: u32 = 3;

fn action_code(action: &TraceAction) -> u32 {
    match action {
        TraceAction::VerifiedClean => VERIFIED,
        TraceAction::Recomputed {
            value_changed: true,
        } => CHANGED,
        TraceAction::Recomputed {
            value_changed: false,
        } => CUTOFF,
    }
}

fn action_of(end: u32) -> TraceAction {
    match (end >> END_BITS) & 3 {
        VERIFIED => TraceAction::VerifiedClean,
        CHANGED => TraceAction::Recomputed {
            value_changed: true,
        },
        _ => TraceAction::Recomputed {
            value_changed: false,
        },
    }
}

fn kind_of(end: u32) -> u32 {
    end >> (END_BITS + 2)
}

fn words_of(kind: u32) -> usize {
    match kind {
        NO_VALUE => 0,
        PAIR => 2,
        _ => 1,
    }
}

struct RefKind {
    type_id: fn() -> TypeId,
    drop: unsafe fn(*mut u8),
}

struct KindOf<V>(PhantomData<V>);

impl<V: 'static> KindOf<V> {
    const KIND: RefKind = RefKind {
        type_id: TypeId::of::<V>,
        drop: drop_ref::<V>,
    };
}

/// # Safety
///
/// `word` holds a `V` that nothing else drops.
unsafe fn drop_ref<V>(word: *mut u8) {
    // SAFETY: the caller's contract.
    unsafe { std::ptr::drop_in_place(word.cast::<V>()) }
}

#[repr(C)]
struct Header {
    refs: AtomicU32,
    generation: u32,
    steps: u32,
    total_nodes: u32,
    kind: &'static RefKind,
    verified_at: AtomicU64,
    changed_at: AtomicU64,
    elapsed_ns: u64,
}

struct Offsets {
    slots: usize,
    gens: usize,
    ends: usize,
    deps: usize,
    words: usize,
    layout: Layout,
}

impl Offsets {
    const ELAPSED: usize = std::mem::size_of::<Header>();

    fn new(steps: usize, deps: usize, words: usize) -> Self {
        let slots = Self::ELAPSED + 8 * steps;
        let gens = slots + 4 * steps;
        let ends = gens + 4 * steps;
        let deps_at = ends + 4 * steps;
        let words_at = (deps_at + 4 * deps).next_multiple_of(8);
        let size = words_at + 8 * words;
        Offsets {
            slots,
            gens,
            ends,
            deps: deps_at,
            words: words_at,
            layout: Layout::from_size_align(size, 8).expect("a kept trace's size overflowed"),
        }
    }
}

/// One step of a [`KeptTrace`], as [`crate::NodeTrace`] has it, with the
/// generation its slot had when the trace was kept and its kept value.
pub struct KeptStep<'a> {
    pub id: NodeId,
    pub generation: u32,
    pub action: TraceAction,
    pub dependencies: &'a [u32],
    pub elapsed_ns: u64,
    pub value: KeptValue<'a>,
}

/// A step's kept value.
pub enum KeptValue<'a> {
    None,
    Count(u64),
    Pair(u64, u64),
    Ref(KeptRef<'a>),
}

/// A kept reference, read back as the type it was kept as.
pub struct KeptRef<'a> {
    word: *const u8,
    kind: &'static RefKind,
    life: PhantomData<&'a KeptTrace>,
}

impl<'a> KeptRef<'a> {
    /// # Panics
    ///
    /// If the trace kept references of another type than `V`.
    pub fn get<V: 'static>(&self) -> &'a V {
        assert!(
            (self.kind.type_id)() == TypeId::of::<V>(),
            "incr-core: a kept trace's value read as another type"
        );
        // SAFETY: the word holds a `V`, checked above, that lives as long
        // as the trace the borrow came from.
        unsafe { &*self.word.cast::<V>() }
    }
}

/// The trace a binding keeps for a `get_traced` root, built by
/// [`crate::Runtime::kept_trace`]. Clones share one allocation, and the
/// kept references drop with the last clone.
pub struct KeptTrace {
    ptr: NonNull<Header>,
}

// SAFETY: the kept references are `Send + Sync` (bounded in `build`), the
// header's mutable fields are atomics, and the rest is never written
// after `build`.
unsafe impl Send for KeptTrace {}
// SAFETY: as for `Send`.
unsafe impl Sync for KeptTrace {}

impl KeptTrace {
    pub(crate) fn build<V: Send + Sync + 'static>(
        generation: u32,
        trace: &PropagationTrace,
        step_generation: impl Fn(u32) -> u32,
        mut value: impl FnMut(&NodeTrace, u32) -> StepValue<V>,
    ) -> KeptTrace {
        const {
            assert!(std::mem::size_of::<V>() <= 8 && std::mem::align_of::<V>() <= 8);
        }
        let steps = trace.node_traces.len();
        let gens: Vec<u32> = trace
            .node_traces
            .iter()
            .map(|t| step_generation(t.id.0))
            .collect();
        let values: Vec<StepValue<V>> = trace
            .node_traces
            .iter()
            .zip(&gens)
            .map(|(t, &generation)| value(t, generation))
            .collect();
        let deps: usize = trace.node_traces.iter().map(|t| t.dependencies.len()).sum();
        let words: usize = values
            .iter()
            .map(|v| match v {
                StepValue::None => 0,
                StepValue::Pair(..) => 2,
                _ => 1,
            })
            .sum();
        assert!(
            deps <= END_MASK as usize && u32::try_from(steps).is_ok(),
            "incr-core: a trace too large to keep"
        );
        let at = Offsets::new(steps, deps, words);
        // SAFETY: the layout has a non-zero size, as it holds the header.
        let base = unsafe { alloc::alloc(at.layout) };
        let Some(ptr) = NonNull::new(base.cast::<Header>()) else {
            alloc::handle_alloc_error(at.layout)
        };
        // SAFETY: every write below lands inside the block, at an offset
        // aligned for its type (the header and `u64` arrays at multiples of
        // eight, `u32` arrays at multiples of four), and the block is
        // eight-aligned.
        unsafe {
            ptr.as_ptr().write(Header {
                refs: AtomicU32::new(1),
                generation,
                steps: steps as u32,
                total_nodes: u32::try_from(trace.total_nodes).unwrap_or(u32::MAX),
                kind: &KindOf::<V>::KIND,
                verified_at: AtomicU64::new(0),
                changed_at: AtomicU64::new(0),
                elapsed_ns: trace.elapsed_ns,
            });
            let mut dep_end = 0usize;
            let mut word = 0usize;
            for (i, (step, value)) in trace.node_traces.iter().zip(values).enumerate() {
                base.add(Offsets::ELAPSED + 8 * i)
                    .cast::<u64>()
                    .write(step.elapsed_ns);
                base.add(at.slots + 4 * i).cast::<u32>().write(step.id.0);
                base.add(at.gens + 4 * i).cast::<u32>().write(gens[i]);
                for dep in &step.dependencies {
                    base.add(at.deps + 4 * dep_end).cast::<u32>().write(dep.0);
                    dep_end += 1;
                }
                let words_at = base.add(at.words + 8 * word);
                let kind = match value {
                    StepValue::None => NO_VALUE,
                    StepValue::Count(n) => {
                        words_at.cast::<u64>().write(n);
                        COUNT
                    }
                    StepValue::Pair(a, b) => {
                        words_at.cast::<u64>().write(a);
                        words_at.add(8).cast::<u64>().write(b);
                        PAIR
                    }
                    StepValue::Ref(v) => {
                        words_at.cast::<V>().write(v);
                        REF
                    }
                };
                word += words_of(kind);
                let end =
                    dep_end as u32 | action_code(&step.action) << END_BITS | kind << (END_BITS + 2);
                base.add(at.ends + 4 * i).cast::<u32>().write(end);
            }
        }
        KeptTrace { ptr }
    }

    fn header(&self) -> &Header {
        // SAFETY: the header lives until the last clone drops.
        unsafe { self.ptr.as_ref() }
    }

    fn base(&self) -> *const u8 {
        self.ptr.as_ptr().cast_const().cast()
    }

    fn u32s(&self, offset: usize, len: usize) -> &[u32] {
        // SAFETY: `offset` is one of the block's `u32` arrays, of `len`
        // initialised entries.
        unsafe { std::slice::from_raw_parts(self.base().add(offset).cast::<u32>(), len) }
    }

    fn ends(&self) -> &[u32] {
        let at = self.offsets_before_words();
        self.u32s(at.ends, self.len())
    }

    fn dep_count(&self) -> usize {
        self.ends()
            .last()
            .map_or(0, |end| (end & END_MASK) as usize)
    }

    fn word_count(&self) -> usize {
        self.ends().iter().map(|&end| words_of(kind_of(end))).sum()
    }

    fn offsets_before_words(&self) -> Offsets {
        Offsets::new(self.len(), 0, 0)
    }

    fn offsets(&self) -> Offsets {
        Offsets::new(self.len(), self.dep_count(), self.word_count())
    }

    /// The number of steps.
    pub fn len(&self) -> usize {
        self.header().steps as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The root's generation when the trace was kept.
    pub fn generation(&self) -> u32 {
        self.header().generation
    }

    /// [`PropagationTrace::total_nodes`], saturated at `u32::MAX`.
    pub fn total_nodes(&self) -> usize {
        self.header().total_nodes as usize
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.header().elapsed_ns
    }

    pub fn nodes_recomputed(&self) -> usize {
        self.steps()
            .filter(|s| matches!(s.action, TraceAction::Recomputed { .. }))
            .count()
    }

    pub fn nodes_cutoff(&self) -> usize {
        self.steps()
            .filter(|s| {
                matches!(
                    s.action,
                    TraceAction::Recomputed {
                        value_changed: false
                    }
                )
            })
            .count()
    }

    /// The steps in trace order.
    pub fn steps(&self) -> impl ExactSizeIterator<Item = KeptStep<'_>> + '_ {
        let at = self.offsets();
        let n = self.len();
        let slots = self.u32s(at.slots, n);
        let gens = self.u32s(at.gens, n);
        let ends = self.u32s(at.ends, n);
        let deps = self.u32s(at.deps, self.dep_count());
        let base = self.base();
        let kind = self.header().kind;
        let mut word = 0usize;
        (0..n).map(move |i| {
            let start = if i == 0 { 0 } else { ends[i - 1] & END_MASK } as usize;
            let end = ends[i];
            // SAFETY: the value words start at `at.words`, and `word` counts
            // the words of the steps before this one.
            let words_at = unsafe { base.add(at.words + 8 * word) };
            let value_kind = kind_of(end);
            word += words_of(value_kind);
            // SAFETY: a count or pair step wrote its `u64` words there.
            let value = unsafe {
                match value_kind {
                    NO_VALUE => KeptValue::None,
                    COUNT => KeptValue::Count(words_at.cast::<u64>().read()),
                    PAIR => KeptValue::Pair(
                        words_at.cast::<u64>().read(),
                        words_at.add(8).cast::<u64>().read(),
                    ),
                    _ => KeptValue::Ref(KeptRef {
                        word: words_at,
                        kind,
                        life: PhantomData,
                    }),
                }
            };
            KeptStep {
                id: NodeId(slots[i]),
                generation: gens[i],
                action: action_of(end),
                dependencies: &deps[start..(end & END_MASK) as usize],
                // SAFETY: step `i`'s `elapsed_ns`.
                elapsed_ns: unsafe { base.add(Offsets::ELAPSED + 8 * i).cast::<u64>().read() },
                value,
            }
        })
    }

    /// The bytes of the allocation.
    pub fn heap_bytes(&self) -> usize {
        self.offsets().layout.size()
    }

    /// The references kept, one per step with a [`StepValue::Ref`].
    pub fn refs(&self) -> usize {
        self.ends()
            .iter()
            .filter(|&&end| kind_of(end) == REF)
            .count()
    }

    pub(crate) fn is_unique(&self) -> bool {
        self.header().refs.load(Ordering::Acquire) == 1
    }

    pub(crate) fn stamps(&self) -> StampPair {
        StampPair {
            verified_at: self.header().verified_at.load(Ordering::Relaxed),
            changed_at: self.header().changed_at.load(Ordering::Relaxed),
        }
    }

    /// Called only under the runtime's inner write lock, which orders it
    /// against the reads of [`Self::stamps`] under that lock.
    pub(crate) fn set_stamps(&self, stamps: StampPair) {
        let h = self.header();
        h.verified_at.store(stamps.verified_at, Ordering::Relaxed);
        h.changed_at.store(stamps.changed_at, Ordering::Relaxed);
    }
}

impl Clone for KeptTrace {
    fn clone(&self) -> Self {
        if self.header().refs.fetch_add(1, Ordering::Relaxed) > i32::MAX as u32 {
            std::process::abort();
        }
        KeptTrace { ptr: self.ptr }
    }
}

impl Drop for KeptTrace {
    fn drop(&mut self) {
        if self.header().refs.fetch_sub(1, Ordering::Release) != 1 {
            return;
        }
        atomic::fence(Ordering::Acquire);
        let at = self.offsets();
        let drop_ref = self.header().kind.drop;
        let base = self.ptr.as_ptr().cast::<u8>();
        let mut word = 0usize;
        for &end in self.ends() {
            if kind_of(end) == REF {
                // SAFETY: this was the last clone, so nothing else reads or
                // drops the reference the step kept there.
                unsafe { drop_ref(base.add(at.words + 8 * word)) };
            }
            word += words_of(kind_of(end));
        }
        // SAFETY: the block was allocated with this layout and this was its
        // last clone.
        unsafe { alloc::dealloc(base, at.layout) };
    }
}

/// What the kept traces of a runtime hold, from
/// [`crate::Runtime::kept_trace_memory`]: `bytes` counts the traces'
/// allocations and the table that maps roots to them, `refs` the
/// references the traces keep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeptTraceMemory {
    pub count: usize,
    pub bytes: usize,
    pub refs: usize,
}

/// Kept traces by root slot, with running totals of their bytes and
/// references.
#[derive(Default)]
pub(crate) struct KeptTraces {
    map: HashMap<u32, KeptTrace>,
    bytes: usize,
    refs: usize,
}

impl KeptTraces {
    pub(crate) fn get(&self, slot: u32) -> Option<&KeptTrace> {
        self.map.get(&slot)
    }

    pub(crate) fn insert(&mut self, slot: u32, kept: KeptTrace) -> Option<KeptTrace> {
        self.bytes += kept.heap_bytes();
        self.refs += kept.refs();
        let old = self.map.insert(slot, kept);
        if let Some(old) = &old {
            self.forget(old);
        }
        old
    }

    pub(crate) fn remove(&mut self, slot: u32) -> Option<KeptTrace> {
        if self.map.is_empty() {
            return None;
        }
        let old = self.map.remove(&slot)?;
        self.forget(&old);
        Some(old)
    }

    fn forget(&mut self, kept: &KeptTrace) {
        self.bytes -= kept.heap_bytes();
        self.refs -= kept.refs();
    }

    pub(crate) fn memory(&self) -> KeptTraceMemory {
        KeptTraceMemory {
            count: self.map.len(),
            bytes: self.bytes + crate::memory::table_bytes::<(u32, KeptTrace)>(self.map.capacity()),
            refs: self.refs,
        }
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.map.capacity()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use super::*;

    fn step(slot: u32, action: TraceAction, deps: &[u32], elapsed_ns: u64) -> NodeTrace {
        NodeTrace {
            id: NodeId(slot),
            action,
            dependencies: deps.iter().map(|&d| NodeId(d)).collect(),
            elapsed_ns,
        }
    }

    const CHANGED_STEP: TraceAction = TraceAction::Recomputed {
        value_changed: true,
    };
    const CUTOFF_STEP: TraceAction = TraceAction::Recomputed {
        value_changed: false,
    };

    fn five_steps() -> PropagationTrace {
        PropagationTrace {
            target: NodeId(9),
            node_traces: vec![
                step(3, CHANGED_STEP, &[], 11),
                step(4, TraceAction::VerifiedClean, &[3], 0),
                step(5, CUTOFF_STEP, &[3, 4], 13),
                step(6, CHANGED_STEP, &[4, 5], u64::MAX),
                step(9, CHANGED_STEP, &[5, 6], 15),
            ],
            total_nodes: 40,
            nodes_recomputed: 4,
            nodes_cutoff: 1,
            elapsed_ns: 1234,
        }
    }

    struct CountDrop(Arc<AtomicUsize>);

    impl Drop for CountDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_kept_trace_reads_back_what_it_was_built_from() {
        let trace = five_steps();
        let values = [
            StepValue::Count(7),
            StepValue::None,
            StepValue::Pair(1, u64::MAX),
            StepValue::Ref(Box::new(60_u32)),
            StepValue::Ref(Box::new(90_u32)),
        ];
        let mut values = values.into_iter();
        let kept = KeptTrace::build(2, &trace, |slot| slot + 100, |_, _| values.next().unwrap());

        assert_eq!(kept.generation(), 2);
        assert_eq!(kept.len(), 5);
        assert_eq!(kept.total_nodes(), 40);
        assert_eq!(kept.elapsed_ns(), 1234);
        assert_eq!(kept.nodes_recomputed(), trace.nodes_recomputed);
        assert_eq!(kept.nodes_cutoff(), trace.nodes_cutoff);
        assert_eq!(kept.refs(), 2);
        assert_eq!(
            kept.stamps(),
            StampPair {
                verified_at: 0,
                changed_at: 0
            }
        );
        for (got, want) in kept.steps().zip(&trace.node_traces) {
            assert_eq!(got.id, want.id);
            assert_eq!(got.generation, want.id.0 + 100);
            assert_eq!(got.action, want.action);
            let deps: Vec<u32> = want.dependencies.iter().map(|d| d.0).collect();
            assert_eq!(got.dependencies, &deps[..]);
            assert_eq!(got.elapsed_ns, want.elapsed_ns);
        }
        let shown: Vec<String> = kept
            .steps()
            .map(|s| match s.value {
                KeptValue::None => "none".to_owned(),
                KeptValue::Count(n) => format!("count {n}"),
                KeptValue::Pair(a, b) => format!("pair {a} {b}"),
                KeptValue::Ref(r) => format!("ref {}", r.get::<Box<u32>>()),
            })
            .collect();
        assert_eq!(
            shown,
            [
                "count 7",
                "none",
                format!("pair 1 {}", u64::MAX).as_str(),
                "ref 60",
                "ref 90"
            ]
        );
        kept.set_stamps(StampPair {
            verified_at: 8,
            changed_at: 6,
        });
        assert_eq!(
            kept.clone().stamps(),
            StampPair {
                verified_at: 8,
                changed_at: 6
            }
        );
    }

    #[test]
    fn a_five_step_trace_with_seven_dependencies_is_one_216_byte_block() {
        let trace = five_steps();
        let kept = KeptTrace::build(0, &trace, |_| 0, |_, _| StepValue::Ref(Box::new(0_u64)));
        assert_eq!(kept.heap_bytes(), 216);
        assert_eq!(std::mem::size_of::<(u32, KeptTrace)>(), 16);
        let bare = KeptTrace::build(0, &trace, |_| 0, |_, _| StepValue::<()>::None);
        assert_eq!(bare.heap_bytes(), 176);
        let empty = PropagationTrace {
            node_traces: Vec::new(),
            ..trace
        };
        let empty = KeptTrace::build(0, &empty, |_| 0, |_, _| StepValue::<()>::None);
        assert_eq!((empty.len(), empty.heap_bytes(), empty.refs()), (0, 48, 0));
        assert_eq!(empty.steps().count(), 0);
    }

    #[test]
    fn kept_refs_drop_once_with_the_last_clone_on_any_thread() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let kept = KeptTrace::build(
            0,
            &five_steps(),
            |_| 0,
            |t, _| match t.id.0 % 2 {
                0 => StepValue::Ref(Arc::new(CountDrop(Arc::clone(&dropped)))),
                _ => StepValue::Count(1),
            },
        );
        assert_eq!(kept.refs(), 2);
        let clones: Vec<KeptTrace> = (0..4).map(|_| kept.clone()).collect();
        drop(kept);
        let threads: Vec<_> = clones
            .into_iter()
            .map(|clone| std::thread::spawn(move || drop(clone)))
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }

    #[test]
    #[should_panic(expected = "read as another type")]
    fn a_kept_ref_read_as_another_type_panics() {
        let kept = KeptTrace::build(
            0,
            &five_steps(),
            |_| 0,
            |_, _| StepValue::Ref(Box::new(1_u32)),
        );
        let first = kept.steps().next().map(|s| s.value);
        if let Some(KeptValue::Ref(r)) = first {
            r.get::<Box<u64>>();
        }
    }

    #[test]
    fn totals_follow_insert_replace_and_remove() {
        let trace = five_steps();
        let build = |refs: bool| {
            KeptTrace::build(
                0,
                &trace,
                |_| 0,
                |_, _| match refs {
                    true => StepValue::Ref(Box::new(0_u8)),
                    false => StepValue::None,
                },
            )
        };
        let mut kept = KeptTraces::default();
        assert_eq!(kept.memory(), KeptTraceMemory::default());
        assert!(kept.remove(1).is_none());
        assert!(kept.insert(1, build(true)).is_none());
        assert!(kept.insert(2, build(false)).is_none());
        let table = table_of(&kept);
        assert_eq!(
            kept.memory(),
            KeptTraceMemory {
                count: 2,
                bytes: 216 + 176 + table,
                refs: 5
            }
        );
        assert!(kept.insert(1, build(false)).is_some());
        assert_eq!(
            kept.memory(),
            KeptTraceMemory {
                count: 2,
                bytes: 176 * 2 + table,
                refs: 0
            }
        );
        assert!(kept.remove(2).is_some());
        assert!(kept.remove(1).is_some());
        assert_eq!(
            kept.memory(),
            KeptTraceMemory {
                count: 0,
                bytes: table,
                refs: 0
            }
        );
    }

    fn table_of(kept: &KeptTraces) -> usize {
        crate::memory::table_bytes::<(u32, KeptTrace)>(kept.capacity())
    }
}
