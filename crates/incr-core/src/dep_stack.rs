//! `DepStack`: strategy-parameterized dependency tracking during compute.
//!
//! When a compute closure runs, every `rt.get(other)` call records `other`
//! as a dependency of the currently-computing node. The recording happens
//! through a per-thread stack of frames: each frame holds the dep set for
//! one nested compute.
//!
//! Shared MUST use thread-local frames. Multiple reader threads can each
//! drive a compute on different nodes simultaneously; if frames lived in
//! a single shared lock, every `rt.get()` would contend on it, killing
//! throughput. The thread-local design lets each thread maintain its
//! own frame stack with no synchronization. Every runtime on the thread
//! shares that stack, so its frames carry their runtime, and the active
//! frame of runtime R is the top frame only if it belongs to R: a read
//! of an R node from another runtime's closure records nothing.
//!
//! Local runtimes share it too: they also nest each other's computes on
//! one thread, and the doom and replay rules read every runtime's frames
//! and their positions on the thread.

use std::cell::RefCell;
use std::mem::ManuallyDrop;

use crate::cycle::{ComputeCtl, BUFFERED, DOOMED, NO_DOOM, POISONED};
use crate::handle::RuntimeId;
use crate::node::NodeId;
use crate::replay::Step;

/// One in-flight compute: the node being computed, its runtime, the deps
/// recorded so far on its behalf, its control block and its position on
/// the conceptual stack.
#[doc(hidden)]
pub struct Frame {
    pub(crate) node: NodeId,
    pub(crate) rt: RuntimeId,
    pub(crate) deps: Vec<(NodeId, u32)>,
    /// Valid while the frame is on the stack: the code that pushed it
    /// owns the control block and pops the frame before it returns.
    pub(crate) ctl: *const ComputeCtl,
    pub(crate) cidx: u32,
    /// The node the frame's closure is blocked reading, as
    /// `slot | generation << 32`, stored by every nested non-Clean read
    /// before it nests: a doom that marks this frame names it.
    pub(crate) reading: u64,
}

impl Frame {
    /// How many times the frame has read the node it reads now, this read
    /// included.
    fn reads_of_reading(&self) -> u32 {
        self.deps
            .iter()
            .filter(|&&(d, g)| (u64::from(d.0) | (u64::from(g) << 32)) == self.reading)
            .count() as u32
    }

    /// The frame's step in the paths of the frames above it.
    pub(crate) fn step(&self) -> Step {
        // SAFETY: the frame is on the stack, so its control block is live
        // (see `Frame::ctl`).
        let member = unsafe { &*self.ctl }.has(BUFFERED);
        Step::of(
            self.rt,
            self.node.0,
            member,
            self.reading,
            self.reads_of_reading(),
        )
    }
}

/// What a nested non-Clean read learns of the frame it is made from.
#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct ActiveRead {
    pub(crate) node: NodeId,
    pub(crate) ctl: *const ComputeCtl,
    pub(crate) cidx: u32,
    /// Frames on the thread's stack, the reading frame included, other
    /// runtimes' frames too.
    pub(crate) depth: u32,
}

impl ActiveRead {
    /// The depth of another runtime's frame reading this runtime's node:
    /// past every budget, so the read takes the cold path, which keeps its
    /// outcome for that frame's retry.
    pub(crate) const FOREIGN: u32 = u32::MAX;
}

/// A stack of frames, innermost last.
#[doc(hidden)]
pub struct Frames {
    pub(crate) frames: Vec<Frame>,
    /// Added to a frame's index to give its conceptual-stack position: a
    /// replay driver runs an entry at its level's position, above frames
    /// whose positions are lower than their indexes would say.
    pub(crate) bias: u32,
    /// The index the frame a driver placed sits at, while it is placed:
    /// the frame below it is not its reader.
    pub(crate) placed: u32,
}

/// A placement [`DepStack::place_next_at`] replaced, to restore.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Placement {
    bias: u32,
    placed: u32,
}

impl Frames {
    #[inline(always)]
    fn active(&mut self, rt: RuntimeId) -> Option<&mut Frame> {
        self.frames.last_mut().filter(|top| top.rt == rt)
    }

    /// A read another runtime serves to the top frame whose outcome no
    /// kept read can carry: that frame can no longer be doomed, because its
    /// retry would read the other runtime again. A frame doomed already
    /// keeps its doom.
    #[cold]
    #[inline(never)]
    fn keep_top_undoomed(&mut self) {
        let top = self
            .frames
            .last()
            .expect("incr-core: a foreign read has a reading frame");
        // SAFETY: the frame is on the stack, so its control block is live
        // (see `Frame::ctl`).
        let ctl = unsafe { &*top.ctl };
        if !ctl.has(DOOMED) {
            ctl.raise(NO_DOOM);
        }
    }

    /// The active frame as the reader of what the compute that popped
    /// above it left: none while a replay driver's placed frame is on top,
    /// since the frame below that one is not its reader.
    #[inline(always)]
    fn reader(&mut self, rt: RuntimeId) -> Option<&mut Frame> {
        if self.frames.len() as u32 == self.placed {
            return None;
        }
        self.active(rt)
    }

    const fn new() -> Self {
        Self {
            frames: Vec::new(),
            bias: 0,
            placed: u32::MAX,
        }
    }

    #[inline(always)]
    fn push(&mut self, frame: Frame) {
        if self.frames.capacity() == 0 {
            first_push();
        }
        self.frames.push(frame);
    }

    #[inline(always)]
    fn frame(node: NodeId, rt: RuntimeId, ctl: &ComputeCtl, cidx: u32) -> Frame {
        Frame {
            node,
            rt,
            deps: Vec::with_capacity(4),
            ctl,
            cidx,
            reading: 0,
        }
    }
}

/// Strategy-parameterized dep tracker used by the runtime during compute.
pub trait DepStack: 'static {
    fn new() -> Self;

    /// Runs `f` on the stack. `f` must not reach user code or the stack.
    #[doc(hidden)]
    fn with<R>(&self, f: impl FnOnce(&mut Frames) -> R) -> R;

    /// Push a frame for `node` of runtime `rt` at the top of the stack,
    /// with the frame's depth as its conceptual-stack position, which is
    /// returned.
    #[doc(hidden)]
    #[inline(always)]
    fn push_frame(&self, rt: RuntimeId, node: NodeId, ctl: &ComputeCtl) -> u32 {
        crate::lock_order::enter_frame();
        self.with(|s| {
            let cidx = (s.frames.len() as u32).wrapping_add(s.bias);
            s.push(Frames::frame(node, rt, ctl, cidx));
            cidx
        })
    }

    /// Push a frame at conceptual-stack position `cidx`: an episode's
    /// member runs all sit at the root's.
    #[doc(hidden)]
    fn push_frame_at(&self, rt: RuntimeId, node: NodeId, ctl: &ComputeCtl, cidx: u32) {
        crate::lock_order::enter_frame();
        self.with(|s| s.push(Frames::frame(node, rt, ctl, cidx)))
    }

    /// Pop the top frame and return the deps it recorded, each with the
    /// generation it was read at.
    #[inline(always)]
    fn pop_frame(&self) -> Vec<(NodeId, u32)> {
        let deps = self.with(|s| {
            s.frames
                .pop()
                .expect("incr-core: pop_frame on an empty stack")
                .deps
        });
        crate::lock_order::leave_frame();
        deps
    }

    /// Record `dep`, read at `generation`, as a dependency of runtime
    /// `rt`'s active frame, if it has one. The generation lets publish
    /// detect a dep that was deleted, and possibly recycled, after the
    /// read. True if the top frame is another runtime's, which records
    /// nothing.
    #[inline(always)]
    fn record_dep(&self, rt: RuntimeId, dep: NodeId, generation: u32) -> bool {
        self.with(|s| match s.frames.last_mut() {
            Some(top) if top.rt == rt => {
                top.deps.push((dep, generation));
                false
            }
            Some(_) => true,
            None => false,
        })
    }

    /// The top frame's runtime, node, control block and position.
    fn top_frame(&self) -> Option<(RuntimeId, NodeId, *const ComputeCtl, u32)> {
        self.with(|s| s.frames.last().map(|f| (f.rt, f.node, f.ctl, f.cidx)))
    }

    /// See [`Frames::keep_top_undoomed`].
    fn keep_top_undoomed(&self) {
        self.with(Frames::keep_top_undoomed);
    }

    /// [`Self::record_dep`] for a nested non-Clean read, which also stores
    /// the read in the frame and returns what the read needs to know of
    /// the frame: for another runtime's frame, at depth
    /// [`ActiveRead::FOREIGN`], with nothing stored.
    #[inline(always)]
    fn record_read(&self, rt: RuntimeId, dep: NodeId, generation: u32) -> Option<ActiveRead> {
        self.with(|s| {
            let depth = s.frames.len() as u32;
            let top = match s.frames.last_mut() {
                Some(top) if top.rt == rt => top,
                Some(top) => {
                    return Some(ActiveRead {
                        node: top.node,
                        ctl: top.ctl,
                        cidx: top.cidx,
                        depth: ActiveRead::FOREIGN,
                    })
                }
                None => return None,
            };
            top.deps.push((dep, generation));
            top.reading = u64::from(dep.0) | (u64::from(generation) << 32);
            Some(ActiveRead {
                node: top.node,
                ctl: top.ctl,
                cidx: top.cidx,
                depth,
            })
        })
    }

    /// The position the next frame pushed with [`Self::push_frame`] gets.
    fn next_cidx(&self) -> u32 {
        self.with(|s| (s.frames.len() as u32).wrapping_add(s.bias))
    }

    /// Makes the next frame pushed with [`Self::push_frame`] sit at
    /// `cidx`, above a virtual frame if `virtual_below`, returning the
    /// placement to restore once that frame has popped.
    fn place_next_at(&self, cidx: u32, virtual_below: bool) -> Placement {
        self.with(|s| {
            let len = s.frames.len() as u32;
            let saved = Placement {
                bias: s.bias,
                placed: s.placed,
            };
            s.bias = cidx.wrapping_sub(len);
            s.placed = if virtual_below { len } else { u32::MAX };
            saved
        })
    }

    fn restore_placement(&self, saved: Placement) {
        self.with(|s| {
            s.bias = saved.bias;
            s.placed = saved.placed;
        });
    }

    /// Mark `rt`'s active frame, if any, as having read a dep that did
    /// not land `Clean`, so its compute must not land `Clean` either.
    fn poison_current_frame(&self, rt: RuntimeId) {
        self.with(|s| {
            if let Some(top) = s.reader(rt) {
                // SAFETY: the frame is on the stack, so its control block
                // is live (see `Frame::ctl`).
                unsafe { (*top.ctl).raise(POISONED) };
            }
        })
    }

    /// `rt`'s active frame as the reader of an unwind out of the compute
    /// that popped above it, with the node it reads and how many times it
    /// has read it.
    fn unwound_reader(&self, rt: RuntimeId) -> Option<(*const ComputeCtl, u32, u64, u32)> {
        self.with(|s| {
            let top = s.reader(rt)?;
            Some((top.ctl, top.cidx, top.reading, top.reads_of_reading()))
        })
    }

    /// `rt`'s active frame's node and position, the node it reads now and
    /// how many times it has read it, this read included.
    fn active_read(&self, rt: RuntimeId) -> Option<(NodeId, u32, u64, u32)> {
        self.with(|s| {
            s.active(rt)
                .map(|top| (top.node, top.cidx, top.reading, top.reads_of_reading()))
        })
    }

    /// How many times `rt`'s active frame has read the node it reads
    /// now, this read included.
    fn reads_of_current(&self, rt: RuntimeId) -> u32 {
        self.with(|s| s.active(rt).map_or(0, |top| top.reads_of_reading()))
    }

    fn active_frame(&self, rt: RuntimeId) -> Option<(*const ComputeCtl, u32)> {
        self.with(|s| s.active(rt).map(|top| (top.ctl, top.cidx)))
    }

    /// The node and control block of `rt`'s active frame, if its top
    /// frame is one of `rt`'s computes. A callback's barrier frame, which
    /// belongs to no runtime, is never one, under either stack.
    fn active_compute(&self, rt: RuntimeId) -> Option<(NodeId, *const ComputeCtl)> {
        self.with(|s| {
            s.frames
                .last()
                .filter(|top| top.rt == rt)
                .map(|top| (top.node, top.ctl))
        })
    }

    /// Whether the thread's top frame is another runtime's than `rt` and
    /// doomed, so a read `rt` serves now comes from a closure that has
    /// read a placeholder.
    fn other_top_doomed(&self, rt: RuntimeId) -> bool {
        self.with(|s| {
            s.frames.last().is_some_and(|top| {
                // SAFETY: the frame is on the stack, so its control block
                // is live (see `Frame::ctl`).
                top.rt != rt && unsafe { &*top.ctl }.has(DOOMED)
            })
        })
    }

    /// True iff `rt` has an active frame (we're inside one of its
    /// compute closures).
    fn current_frame_active(&self, rt: RuntimeId) -> bool {
        self.with(|s| s.active(rt).is_some())
    }

    /// True iff any runtime's compute frame is on this thread's stack,
    /// at any depth. A callback's barrier frame belongs to no runtime and
    /// does not count.
    fn any_compute_frame(&self) -> bool {
        self.with(|s| s.frames.iter().any(|f| f.rt != RuntimeId::SENTINEL))
    }

    /// The depth of the innermost frame on this stack computing `node`
    /// of runtime `rt`, if one is. Depth 0 is the outermost frame.
    fn frame_depth(&self, rt: RuntimeId, node: NodeId) -> Option<usize> {
        self.with(|s| s.frames.iter().rposition(|f| f.node == node && f.rt == rt))
    }

    /// The position of the innermost frame on this stack computing
    /// `node` of runtime `rt`, if one is.
    fn frame_cidx(&self, rt: RuntimeId, node: NodeId) -> Option<u32> {
        self.with(|s| {
            s.frames
                .iter()
                .rev()
                .find(|f| f.node == node && f.rt == rt)
                .map(|f| f.cidx)
        })
    }

    /// True iff `node` of runtime `rt` is being computed by a frame on
    /// this stack.
    fn is_computing(&self, rt: RuntimeId, node: NodeId) -> bool {
        self.frame_depth(rt, node).is_some()
    }

    /// The number of frames on the stack; right after a pop, the popped
    /// frame's depth.
    fn depth(&self) -> usize {
        self.with(|s| s.frames.len())
    }
}

/// A Local runtime's view of its thread's stack, whose address it takes
/// once, at creation. The pointer keeps it `!Send`, as `Runtime<Local>`
/// is:
///
/// ```compile_fail,E0277
/// fn send<T: Send>() {}
/// send::<incr_core::runtime::Runtime<incr_core::cells::Local>>();
/// ```
pub struct LocalDepStack {
    frames: *const RefCell<Frames>,
}

impl DepStack for LocalDepStack {
    fn new() -> Self {
        Self {
            frames: SHARED_FRAMES.with(|f| std::ptr::from_ref(&**f)),
        }
    }

    #[inline(always)]
    fn with<R>(&self, f: impl FnOnce(&mut Frames) -> R) -> R {
        // SAFETY: the pointer was taken on this thread, the only one a
        // Local runtime runs on, and `SHARED_FRAMES` has no destructor,
        // so its storage lasts as long as the thread. The reference does
        // not outlive this call.
        f(&mut unsafe { &*self.frames }.borrow_mut())
    }
}

pub struct SharedDepStack;

thread_local! {
    /// Without a destructor, so an access needs no check of the
    /// thread-local's state; `FRAMES_GUARD` frees the buffer instead.
    static SHARED_FRAMES: ManuallyDrop<RefCell<Frames>> =
        const { ManuallyDrop::new(RefCell::new(Frames::new())) };
    static FRAMES_GUARD: FramesGuard = const { FramesGuard };
}

/// Frees this thread's frame buffer as the thread exits, and leaves an
/// empty stack in `SHARED_FRAMES` for any later access.
struct FramesGuard;

impl Drop for FramesGuard {
    fn drop(&mut self) {
        SHARED_FRAMES.with(|f| {
            if let Ok(mut frames) = f.try_borrow_mut() {
                drop(std::mem::replace(&mut *frames, Frames::new()));
            }
        });
    }
}

/// Registers `FRAMES_GUARD`'s destructor, on the push that allocates the
/// thread's buffer. During thread-local destruction the registration is
/// refused, and that late buffer is leaked.
#[cold]
#[inline(never)]
fn first_push() {
    let _ = FRAMES_GUARD.try_with(|_| {});
}

impl DepStack for SharedDepStack {
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn with<R>(&self, f: impl FnOnce(&mut Frames) -> R) -> R {
        // Only the address is taken inside `LocalKey::with`, so that call
        // stays small enough to inline at every site; with `f` inside it,
        // the compiler leaves it out of line on the `get` path.
        let frames: *const RefCell<Frames> = SHARED_FRAMES.with(|f| std::ptr::from_ref(&**f));
        // SAFETY: a thread's locals are destroyed only as the thread
        // exits, never during a call on that thread, and the reference
        // does not outlive this call.
        f(&mut unsafe { &*frames }.borrow_mut())
    }
}

#[cfg(loom)]
pub struct LoomDepStack;

#[cfg(loom)]
loom::thread_local! {
    static LOOM_FRAMES: RefCell<Frames> = RefCell::new(Frames::new());
}

#[cfg(loom)]
impl DepStack for LoomDepStack {
    fn new() -> Self {
        Self
    }

    fn with<R>(&self, f: impl FnOnce(&mut Frames) -> R) -> R {
        LOOM_FRAMES.with(|s| f(&mut s.borrow_mut()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(n: u64) -> RuntimeId {
        RuntimeId::for_test(n)
    }

    #[test]
    fn push_pop_records() {
        let s = SharedDepStack::new();
        let ctl = ComputeCtl::new(0);
        assert!(!s.current_frame_active(rt(1)));
        assert_eq!(s.push_frame(rt(1), NodeId(100), &ctl), 0);
        assert!(s.current_frame_active(rt(1)));
        s.record_dep(rt(1), NodeId(0), 0);
        s.record_dep(rt(1), NodeId(1), 0);
        s.record_dep(rt(1), NodeId(2), 0);
        let frame = s.pop_frame();
        assert_eq!(frame, vec![(NodeId(0), 0), (NodeId(1), 0), (NodeId(2), 0)]);
        assert!(!s.current_frame_active(rt(1)));
    }

    #[test]
    fn nested_frames_are_independent() {
        let s = SharedDepStack::new();
        let (a, b) = (ComputeCtl::new(0), ComputeCtl::new(0));
        s.push_frame(rt(1), NodeId(100), &a);
        s.record_dep(rt(1), NodeId(1), 0);
        assert_eq!(s.push_frame(rt(1), NodeId(101), &b), 1);
        s.record_dep(rt(1), NodeId(2), 0);
        s.record_dep(rt(1), NodeId(3), 0);
        assert_eq!(s.pop_frame(), vec![(NodeId(2), 0), (NodeId(3), 0)]);
        assert_eq!(s.pop_frame(), vec![(NodeId(1), 0)]);
    }

    #[test]
    fn poison_marks_only_the_top_frame() {
        let s = SharedDepStack::new();
        let (a, b) = (ComputeCtl::new(0), ComputeCtl::new(0));
        s.push_frame(rt(1), NodeId(100), &a);
        s.push_frame(rt(1), NodeId(101), &b);
        s.poison_current_frame(rt(1));
        s.pop_frame();
        s.pop_frame();
        assert!(b.has(POISONED));
        assert!(!a.has(POISONED));
        s.poison_current_frame(rt(1));
    }

    #[test]
    fn poison_skips_a_frame_below_a_placed_position() {
        let s = SharedDepStack::new();
        let a = ComputeCtl::new(0);
        s.push_frame(rt(1), NodeId(100), &a);
        let placement = s.place_next_at(3, true);
        s.poison_current_frame(rt(1));
        assert!(!a.has(POISONED));
        s.restore_placement(placement);
        s.poison_current_frame(rt(1));
        assert!(a.has(POISONED));
        s.pop_frame();
    }

    #[test]
    fn record_outside_frame_is_noop() {
        let s = SharedDepStack::new();
        s.record_dep(rt(1), NodeId(42), 0);
        assert!(!s.current_frame_active(rt(1)));
    }

    #[test]
    fn is_computing_tracks_open_frames() {
        let s = SharedDepStack::new();
        let (a, b) = (ComputeCtl::new(0), ComputeCtl::new(0));
        assert!(!s.is_computing(rt(1), NodeId(7)));
        s.push_frame(rt(1), NodeId(7), &a);
        s.push_frame(rt(1), NodeId(8), &b);
        assert!(s.is_computing(rt(1), NodeId(7)));
        assert!(s.is_computing(rt(1), NodeId(8)));
        assert!(!s.is_computing(rt(1), NodeId(9)));
        s.pop_frame();
        assert!(!s.is_computing(rt(1), NodeId(8)));
        assert!(s.is_computing(rt(1), NodeId(7)));
        s.pop_frame();
        assert!(!s.is_computing(rt(1), NodeId(7)));
    }

    #[test]
    fn frames_are_scoped_to_their_runtime() {
        let s = SharedDepStack::new();
        let (a, b) = (ComputeCtl::new(0), ComputeCtl::new(0));
        s.push_frame(rt(1), NodeId(5), &a);
        s.push_frame(rt(2), NodeId(5), &b);
        assert!(!s.current_frame_active(rt(1)));
        s.record_dep(rt(1), NodeId(9), 0);
        s.poison_current_frame(rt(1));
        assert!(!b.has(POISONED));
        assert_eq!(s.frame_depth(rt(1), NodeId(5)), Some(0));
        assert_eq!(s.frame_depth(rt(2), NodeId(5)), Some(1));
        assert!(s.pop_frame().is_empty());
        assert!(s.current_frame_active(rt(1)));
        s.pop_frame();
    }

    #[test]
    fn any_compute_frame_sees_every_runtime_but_not_a_barrier() {
        let s = SharedDepStack::new();
        let (a, b) = (ComputeCtl::new(0), ComputeCtl::new(0));
        assert!(!s.any_compute_frame());
        s.push_frame_at(RuntimeId::SENTINEL, NodeId::SENTINEL, &a, u32::MAX);
        assert!(!s.any_compute_frame());
        s.pop_frame();
        s.push_frame(rt(2), NodeId(5), &a);
        s.push_frame_at(RuntimeId::SENTINEL, NodeId::SENTINEL, &b, u32::MAX);
        assert!(!s.current_frame_active(rt(1)));
        assert!(s.any_compute_frame(), "a frame below the barrier counts");
        s.pop_frame();
        s.pop_frame();
        assert!(!s.any_compute_frame());
    }

    #[test]
    fn local_runtimes_share_the_threads_frames() {
        let (s1, s2) = (LocalDepStack::new(), LocalDepStack::new());
        let (a, b) = (ComputeCtl::new(0), ComputeCtl::new(0));
        s1.push_frame(rt(1), NodeId(5), &a);
        assert_eq!(s2.push_frame(rt(2), NodeId(6), &b), 1);
        assert!(!s1.current_frame_active(rt(1)));
        s1.record_dep(rt(1), NodeId(9), 0);
        s1.poison_current_frame(rt(1));
        assert!(!a.has(POISONED));
        assert!(!s1.other_top_doomed(rt(1)));
        b.raise(DOOMED);
        assert!(s1.other_top_doomed(rt(1)));
        assert!(!s2.other_top_doomed(rt(2)));
        assert!(s2.pop_frame().is_empty());
        assert!(s1.current_frame_active(rt(1)));
        s1.record_dep(rt(1), NodeId(10), 0);
        assert_eq!(s1.pop_frame(), vec![(NodeId(10), 0)]);
    }

    #[test]
    fn is_computing_is_thread_local() {
        let s = SharedDepStack::new();
        let ctl = ComputeCtl::new(0);
        s.push_frame(rt(1), NodeId(5), &ctl);
        assert!(s.is_computing(rt(1), NodeId(5)));
        std::thread::spawn(|| {
            let s2 = SharedDepStack::new();
            assert!(!s2.is_computing(rt(1), NodeId(5)));
        })
        .join()
        .unwrap();
        s.pop_frame();
    }
}
