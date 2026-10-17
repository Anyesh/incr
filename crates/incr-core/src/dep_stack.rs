//! `DepStack`: strategy-parameterized dependency tracking during compute.
//!
//! When a compute closure runs, every `rt.get(other)` call records `other`
//! as a dependency of the currently-computing node. The recording happens
//! through a per-thread (Shared) or per-runtime (Local) stack of frames:
//! each frame holds the dep set for one nested compute. The stack handles
//! nested computes that may happen during operator evaluation.
//!
//! Shared MUST use thread-local frames. Multiple reader threads can each
//! drive a compute on different nodes simultaneously; if frames lived in
//! a single shared lock, every `rt.get()` would contend on it, killing
//! throughput. The thread-local design lets each thread maintain its
//! own frame stack with no synchronization.
//!
//! Local uses a RefCell-backed stack because there's only one thread.
//! RefCell's borrow counter is cheaper than a thread_local key lookup.
//!
//! The trade-off for Shared's thread_local: two `Runtime<Shared>`
//! instances on the same thread share the same frame stack, so nesting
//! computes across runtimes would mix them. This is the same constraint
//! the production incr-concurrent already imposes. Tests use one
//! runtime; production users should treat the runtime as a singleton
//! per logical concern.

use std::cell::RefCell;

use crate::node::NodeId;

/// Strategy-parameterized dep tracker used by the runtime during compute.
pub trait DepStack: 'static {
    fn new() -> Self;

    /// Push a fresh frame at the top of the stack. Called when entering
    /// a compute closure; `node` is the node being computed so cycle
    /// detection can ask whether a given node is already on this
    /// thread's compute stack.
    fn push_frame(&self, node: NodeId);

    /// Pop the top frame and return what it recorded. Called when exiting
    /// a compute closure.
    fn pop_frame(&self) -> PoppedFrame;

    /// Mark the top frame (if any) as having read a dep that did not land
    /// `Clean`, so its compute must not land `Clean` either.
    fn poison_current_frame(&self);

    /// Record `dep`, read at `generation`, as a dependency of the
    /// currently-computing node. Called by `rt.get()` whenever a frame is
    /// active. No-op when no frame is active (e.g., a top-level `get` from
    /// user code). The generation lets publish detect a dep that was
    /// deleted, and possibly recycled, after the read.
    fn record_dep(&self, dep: NodeId, generation: u32);

    /// True iff at least one frame is active (i.e., we're inside a
    /// compute closure).
    fn current_frame_active(&self) -> bool;

    /// True iff `node` is currently being computed by THIS thread
    /// (Shared) or this runtime (Local). Observing `Computing` on such a
    /// node means the dependency graph has a cycle: the only way to
    /// re-reach a node we are mid-computing is through its own deps.
    fn is_computing(&self, node: NodeId) -> bool;

    /// The depth of this thread's (Shared) or this runtime's (Local)
    /// frame that is computing `node`, if one is. Depth 0 is the
    /// outermost frame.
    fn frame_depth(&self, node: NodeId) -> Option<usize>;

    /// Record that the frame at `depth` owns taint memo entries, so its
    /// pop reports `memo_owner` and the owner clears them. The flag lives
    /// on the frame so a pop that owns nothing tests nothing else.
    fn mark_memo_owner(&self, depth: usize);
}

/// What a finished compute frame recorded.
#[derive(Debug, PartialEq, Eq)]
pub struct PoppedFrame {
    /// Each dep read, with the generation it was read at.
    pub deps: Vec<(NodeId, u32)>,
    /// True if a dep read inside the frame did not land `Clean`.
    pub poisoned: bool,
    /// The frame's depth; 0 is the outermost frame.
    pub depth: u32,
    /// True if the frame owns taint memo entries (`mark_memo_owner`).
    pub memo_owner: bool,
}

/// One in-flight compute: the node being computed and the deps recorded
/// so far on its behalf.
struct Frame {
    node: NodeId,
    deps: Vec<(NodeId, u32)>,
    poisoned: bool,
    memo_owner: bool,
}

impl Frame {
    fn new(node: NodeId) -> Self {
        Self {
            node,
            deps: Vec::with_capacity(4),
            poisoned: false,
            memo_owner: false,
        }
    }

    fn finish(self, depth: usize) -> PoppedFrame {
        PoppedFrame {
            deps: self.deps,
            poisoned: self.poisoned,
            depth: depth as u32,
            memo_owner: self.memo_owner,
        }
    }
}

fn pop(frames: &mut Vec<Frame>, what: &str) -> PoppedFrame {
    let frame = frames
        .pop()
        .unwrap_or_else(|| panic!("{what}::pop_frame on empty stack"));
    frame.finish(frames.len())
}

/// Local strategy: RefCell-backed stack on the runtime.
pub struct LocalDepStack {
    stack: RefCell<Vec<Frame>>,
}

impl DepStack for LocalDepStack {
    fn new() -> Self {
        Self {
            stack: RefCell::new(Vec::new()),
        }
    }

    fn push_frame(&self, node: NodeId) {
        self.stack.borrow_mut().push(Frame::new(node));
    }

    fn pop_frame(&self) -> PoppedFrame {
        pop(&mut self.stack.borrow_mut(), "LocalDepStack")
    }

    fn record_dep(&self, dep: NodeId, generation: u32) {
        let mut frames = self.stack.borrow_mut();
        if let Some(frame) = frames.last_mut() {
            frame.deps.push((dep, generation));
        }
    }

    fn poison_current_frame(&self) {
        if let Some(frame) = self.stack.borrow_mut().last_mut() {
            frame.poisoned = true;
        }
    }

    fn current_frame_active(&self) -> bool {
        !self.stack.borrow().is_empty()
    }

    fn is_computing(&self, node: NodeId) -> bool {
        self.stack.borrow().iter().any(|f| f.node == node)
    }

    fn frame_depth(&self, node: NodeId) -> Option<usize> {
        self.stack.borrow().iter().position(|f| f.node == node)
    }

    fn mark_memo_owner(&self, depth: usize) {
        self.stack.borrow_mut()[depth].memo_owner = true;
    }
}

/// Shared strategy: thread-local stack. The `SharedDepStack` value
/// itself carries no state; it only routes calls to the thread_local.
///
/// Limitation: two `Runtime<Shared>` instances on the same thread share
/// the same stack. Don't nest one runtime's compute inside another's
/// on the same thread.
pub struct SharedDepStack;

thread_local! {
    static SHARED_FRAMES: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
}

impl DepStack for SharedDepStack {
    fn new() -> Self {
        Self
    }

    fn push_frame(&self, node: NodeId) {
        SHARED_FRAMES.with(|f| f.borrow_mut().push(Frame::new(node)));
    }

    fn pop_frame(&self) -> PoppedFrame {
        SHARED_FRAMES.with(|f| pop(&mut f.borrow_mut(), "SharedDepStack"))
    }

    fn record_dep(&self, dep: NodeId, generation: u32) {
        SHARED_FRAMES.with(|f| {
            let mut frames = f.borrow_mut();
            if let Some(frame) = frames.last_mut() {
                frame.deps.push((dep, generation));
            }
        });
    }

    fn poison_current_frame(&self) {
        SHARED_FRAMES.with(|f| {
            if let Some(frame) = f.borrow_mut().last_mut() {
                frame.poisoned = true;
            }
        });
    }

    fn current_frame_active(&self) -> bool {
        SHARED_FRAMES.with(|f| !f.borrow().is_empty())
    }

    fn is_computing(&self, node: NodeId) -> bool {
        SHARED_FRAMES.with(|f| f.borrow().iter().any(|fr| fr.node == node))
    }

    fn frame_depth(&self, node: NodeId) -> Option<usize> {
        SHARED_FRAMES.with(|f| f.borrow().iter().position(|fr| fr.node == node))
    }

    fn mark_memo_owner(&self, depth: usize) {
        SHARED_FRAMES.with(|f| f.borrow_mut()[depth].memo_owner = true);
    }
}

/// Loom strategy: the same per-thread stack as [`SharedDepStack`], on
/// `loom::thread_local!`. Loom runs every modeled thread on one OS
/// thread, so a `std::thread_local!` stack would be shared by all of
/// them and mix their frames.
#[cfg(loom)]
pub struct LoomDepStack;

#[cfg(loom)]
loom::thread_local! {
    static LOOM_FRAMES: RefCell<Vec<Frame>> = RefCell::new(Vec::new());
}

#[cfg(loom)]
impl DepStack for LoomDepStack {
    fn new() -> Self {
        Self
    }

    fn push_frame(&self, node: NodeId) {
        LOOM_FRAMES.with(|f| f.borrow_mut().push(Frame::new(node)));
    }

    fn pop_frame(&self) -> PoppedFrame {
        LOOM_FRAMES.with(|f| pop(&mut f.borrow_mut(), "LoomDepStack"))
    }

    fn record_dep(&self, dep: NodeId, generation: u32) {
        LOOM_FRAMES.with(|f| {
            if let Some(frame) = f.borrow_mut().last_mut() {
                frame.deps.push((dep, generation));
            }
        });
    }

    fn poison_current_frame(&self) {
        LOOM_FRAMES.with(|f| {
            if let Some(frame) = f.borrow_mut().last_mut() {
                frame.poisoned = true;
            }
        });
    }

    fn current_frame_active(&self) -> bool {
        LOOM_FRAMES.with(|f| !f.borrow().is_empty())
    }

    fn is_computing(&self, node: NodeId) -> bool {
        LOOM_FRAMES.with(|f| f.borrow().iter().any(|fr| fr.node == node))
    }

    fn frame_depth(&self, node: NodeId) -> Option<usize> {
        LOOM_FRAMES.with(|f| f.borrow().iter().position(|fr| fr.node == node))
    }

    fn mark_memo_owner(&self, depth: usize) {
        LOOM_FRAMES.with(|f| f.borrow_mut()[depth].memo_owner = true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_push_pop_records() {
        let s = LocalDepStack::new();
        assert!(!s.current_frame_active());
        s.push_frame(NodeId(100));
        assert!(s.current_frame_active());
        s.record_dep(NodeId(0), 0);
        s.record_dep(NodeId(1), 0);
        s.record_dep(NodeId(2), 0);
        let frame = s.pop_frame().deps;
        assert_eq!(frame, vec![(NodeId(0), 0), (NodeId(1), 0), (NodeId(2), 0)]);
        assert!(!s.current_frame_active());
    }

    #[test]
    fn local_nested_frames_are_independent() {
        let s = LocalDepStack::new();
        s.push_frame(NodeId(100));
        s.record_dep(NodeId(1), 0);
        s.push_frame(NodeId(101));
        s.record_dep(NodeId(2), 0);
        s.record_dep(NodeId(3), 0);
        let inner = s.pop_frame().deps;
        assert_eq!(inner, vec![(NodeId(2), 0), (NodeId(3), 0)]);
        let outer = s.pop_frame().deps;
        assert_eq!(outer, vec![(NodeId(1), 0)]);
    }

    #[test]
    fn local_poison_marks_only_the_top_frame() {
        let s = LocalDepStack::new();
        s.push_frame(NodeId(100));
        s.push_frame(NodeId(101));
        s.poison_current_frame();
        assert!(s.pop_frame().poisoned);
        assert!(!s.pop_frame().poisoned);
        s.poison_current_frame();
    }

    #[test]
    fn local_record_outside_frame_is_noop() {
        let s = LocalDepStack::new();
        s.record_dep(NodeId(42), 0);
        assert!(!s.current_frame_active());
    }

    #[test]
    fn local_is_computing_tracks_open_frames() {
        let s = LocalDepStack::new();
        assert!(!s.is_computing(NodeId(7)));
        s.push_frame(NodeId(7));
        s.push_frame(NodeId(8));
        assert!(s.is_computing(NodeId(7)));
        assert!(s.is_computing(NodeId(8)));
        assert!(!s.is_computing(NodeId(9)));
        s.pop_frame();
        assert!(!s.is_computing(NodeId(8)));
        assert!(s.is_computing(NodeId(7)));
        s.pop_frame();
        assert!(!s.is_computing(NodeId(7)));
    }

    #[test]
    fn shared_push_pop_records() {
        let s = SharedDepStack::new();
        s.push_frame(NodeId(100));
        s.record_dep(NodeId(7), 0);
        s.record_dep(NodeId(11), 0);
        let frame = s.pop_frame().deps;
        assert_eq!(frame, vec![(NodeId(7), 0), (NodeId(11), 0)]);
    }

    #[test]
    fn shared_is_computing_is_thread_local() {
        let s = SharedDepStack::new();
        s.push_frame(NodeId(5));
        assert!(s.is_computing(NodeId(5)));
        std::thread::spawn(|| {
            let s2 = SharedDepStack::new();
            assert!(!s2.is_computing(NodeId(5)));
        })
        .join()
        .unwrap();
        s.pop_frame();
    }
}
