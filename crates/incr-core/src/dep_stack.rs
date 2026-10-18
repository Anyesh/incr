//! `DepStack`: strategy-parameterized dependency tracking during compute.
//!
//! When a compute closure runs, every `rt.get(other)` call records `other`
//! as a dependency of the currently-computing node. The recording happens
//! through a per-thread (Shared) or per-runtime (Local) stack of frames:
//! each frame holds the dep set for one nested compute.
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
//! Local uses a RefCell-backed stack per runtime because there's only
//! one thread. RefCell's borrow counter is cheaper than a thread_local
//! key lookup.

use std::cell::RefCell;

use crate::cycle::{ComputeCtl, POISONED};
use crate::handle::RuntimeId;
use crate::node::NodeId;

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
}

/// A stack of frames, innermost last.
#[doc(hidden)]
pub struct Frames {
    pub(crate) frames: Vec<Frame>,
}

impl Frames {
    #[inline(always)]
    fn active(&mut self, rt: RuntimeId, thread_wide: bool) -> Option<&mut Frame> {
        self.frames
            .last_mut()
            .filter(|top| !thread_wide || top.rt == rt)
    }

    const fn new() -> Self {
        Self { frames: Vec::new() }
    }
}

/// Strategy-parameterized dep tracker used by the runtime during compute.
pub trait DepStack: 'static {
    /// Every runtime on the thread shares the stack, so the active frame
    /// of a runtime is the top frame only if it belongs to it.
    const THREAD_WIDE: bool;

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
        self.with(|s| {
            let cidx = s.frames.len() as u32;
            s.frames.push(Frame {
                node,
                rt,
                deps: Vec::with_capacity(4),
                ctl,
                cidx,
            });
            cidx
        })
    }

    /// Push a frame at conceptual-stack position `cidx`: an episode's
    /// member runs all sit at the root's.
    #[doc(hidden)]
    fn push_frame_at(&self, rt: RuntimeId, node: NodeId, ctl: &ComputeCtl, cidx: u32) {
        self.with(|s| {
            s.frames.push(Frame {
                node,
                rt,
                deps: Vec::with_capacity(4),
                ctl,
                cidx,
            })
        })
    }

    /// Pop the top frame and return the deps it recorded, each with the
    /// generation it was read at.
    #[inline(always)]
    fn pop_frame(&self) -> Vec<(NodeId, u32)> {
        self.with(|s| {
            s.frames
                .pop()
                .expect("incr-core: pop_frame on an empty stack")
                .deps
        })
    }

    /// Record `dep`, read at `generation`, as a dependency of runtime
    /// `rt`'s active frame, if it has one. The generation lets publish
    /// detect a dep that was deleted, and possibly recycled, after the
    /// read.
    #[inline(always)]
    fn record_dep(&self, rt: RuntimeId, dep: NodeId, generation: u32) {
        self.with(|s| {
            if let Some(top) = s.active(rt, Self::THREAD_WIDE) {
                top.deps.push((dep, generation));
            }
        })
    }

    /// Mark `rt`'s active frame, if any, as having read a dep that did
    /// not land `Clean`, so its compute must not land `Clean` either.
    fn poison_current_frame(&self, rt: RuntimeId) {
        self.with(|s| {
            if let Some(top) = s.active(rt, Self::THREAD_WIDE) {
                // SAFETY: the frame is on the stack, so its control block
                // is live (see `Frame::ctl`).
                unsafe { (*top.ctl).raise(POISONED) };
            }
        })
    }

    /// The control block and position of `rt`'s active frame.
    #[doc(hidden)]
    fn active_frame(&self, rt: RuntimeId) -> Option<(*const ComputeCtl, u32)> {
        self.with(|s| {
            s.active(rt, Self::THREAD_WIDE)
                .map(|top| (top.ctl, top.cidx))
        })
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

    /// True iff `rt` has an active frame (we're inside one of its
    /// compute closures).
    fn current_frame_active(&self, rt: RuntimeId) -> bool {
        self.with(|s| s.active(rt, Self::THREAD_WIDE).is_some())
    }

    /// The depth of the innermost frame on this stack computing `node`
    /// of runtime `rt`, if one is. Depth 0 is the outermost frame.
    fn frame_depth(&self, rt: RuntimeId, node: NodeId) -> Option<usize> {
        self.with(|s| {
            s.frames
                .iter()
                .rposition(|f| f.node == node && (!Self::THREAD_WIDE || f.rt == rt))
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

pub struct LocalDepStack {
    stack: RefCell<Frames>,
}

impl DepStack for LocalDepStack {
    const THREAD_WIDE: bool = false;

    fn new() -> Self {
        Self {
            stack: RefCell::new(Frames::new()),
        }
    }

    #[inline(always)]
    fn with<R>(&self, f: impl FnOnce(&mut Frames) -> R) -> R {
        f(&mut self.stack.borrow_mut())
    }
}

pub struct SharedDepStack;

thread_local! {
    static SHARED_FRAMES: RefCell<Frames> = const { RefCell::new(Frames::new()) };
}

impl DepStack for SharedDepStack {
    const THREAD_WIDE: bool = true;

    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn with<R>(&self, f: impl FnOnce(&mut Frames) -> R) -> R {
        // Only the address is taken inside `LocalKey::with`, so that call
        // stays small enough to inline at every site; with `f` inside it,
        // the compiler leaves it out of line on the `get` path.
        let frames: *const RefCell<Frames> = SHARED_FRAMES.with(std::ptr::from_ref);
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
    const THREAD_WIDE: bool = true;

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
    fn local_push_pop_records() {
        let s = LocalDepStack::new();
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
    fn local_nested_frames_are_independent() {
        let s = LocalDepStack::new();
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
    fn local_poison_marks_only_the_top_frame() {
        let s = LocalDepStack::new();
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
    fn local_record_outside_frame_is_noop() {
        let s = LocalDepStack::new();
        s.record_dep(rt(1), NodeId(42), 0);
        assert!(!s.current_frame_active(rt(1)));
    }

    #[test]
    fn local_is_computing_tracks_open_frames() {
        let s = LocalDepStack::new();
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
    fn shared_push_pop_records() {
        let s = SharedDepStack::new();
        let ctl = ComputeCtl::new(0);
        s.push_frame(rt(1), NodeId(100), &ctl);
        s.record_dep(rt(1), NodeId(7), 0);
        s.record_dep(rt(1), NodeId(11), 0);
        assert_eq!(s.pop_frame(), vec![(NodeId(7), 0), (NodeId(11), 0)]);
    }

    #[test]
    fn shared_frames_are_scoped_to_their_runtime() {
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
    fn shared_is_computing_is_thread_local() {
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
