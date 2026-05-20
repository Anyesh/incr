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
    /// a compute closure.
    fn push_frame(&self);

    /// Pop the top frame and return its recorded deps. Called when
    /// exiting a compute closure.
    fn pop_frame(&self) -> Vec<NodeId>;

    /// Record `dep` as a dependency of the currently-computing node. Called
    /// by `rt.get()` whenever a frame is active. No-op when no frame is
    /// active (e.g., a top-level `get` from user code).
    fn record_dep(&self, dep: NodeId);

    /// True iff at least one frame is active (i.e., we're inside a
    /// compute closure).
    fn current_frame_active(&self) -> bool;
}

/// Local strategy: RefCell-backed stack on the runtime.
pub struct LocalDepStack {
    stack: RefCell<Vec<Vec<NodeId>>>,
}

impl DepStack for LocalDepStack {
    fn new() -> Self {
        Self {
            stack: RefCell::new(Vec::new()),
        }
    }

    fn push_frame(&self) {
        self.stack.borrow_mut().push(Vec::with_capacity(4));
    }

    fn pop_frame(&self) -> Vec<NodeId> {
        self.stack
            .borrow_mut()
            .pop()
            .expect("LocalDepStack::pop_frame on empty stack")
    }

    fn record_dep(&self, dep: NodeId) {
        let mut frames = self.stack.borrow_mut();
        if let Some(frame) = frames.last_mut() {
            frame.push(dep);
        }
    }

    fn current_frame_active(&self) -> bool {
        !self.stack.borrow().is_empty()
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
    static SHARED_FRAMES: RefCell<Vec<Vec<NodeId>>> = const { RefCell::new(Vec::new()) };
}

impl DepStack for SharedDepStack {
    fn new() -> Self {
        Self
    }

    fn push_frame(&self) {
        SHARED_FRAMES.with(|f| f.borrow_mut().push(Vec::with_capacity(4)));
    }

    fn pop_frame(&self) -> Vec<NodeId> {
        SHARED_FRAMES.with(|f| {
            f.borrow_mut()
                .pop()
                .expect("SharedDepStack::pop_frame on empty stack")
        })
    }

    fn record_dep(&self, dep: NodeId) {
        SHARED_FRAMES.with(|f| {
            let mut frames = f.borrow_mut();
            if let Some(frame) = frames.last_mut() {
                frame.push(dep);
            }
        });
    }

    fn current_frame_active(&self) -> bool {
        SHARED_FRAMES.with(|f| !f.borrow().is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_push_pop_records() {
        let s = LocalDepStack::new();
        assert!(!s.current_frame_active());
        s.push_frame();
        assert!(s.current_frame_active());
        s.record_dep(NodeId(0));
        s.record_dep(NodeId(1));
        s.record_dep(NodeId(2));
        let frame = s.pop_frame();
        assert_eq!(frame, vec![NodeId(0), NodeId(1), NodeId(2)]);
        assert!(!s.current_frame_active());
    }

    #[test]
    fn local_nested_frames_are_independent() {
        let s = LocalDepStack::new();
        s.push_frame();
        s.record_dep(NodeId(1));
        s.push_frame();
        s.record_dep(NodeId(2));
        s.record_dep(NodeId(3));
        let inner = s.pop_frame();
        assert_eq!(inner, vec![NodeId(2), NodeId(3)]);
        let outer = s.pop_frame();
        assert_eq!(outer, vec![NodeId(1)]);
    }

    #[test]
    fn local_record_outside_frame_is_noop() {
        let s = LocalDepStack::new();
        s.record_dep(NodeId(42));
        assert!(!s.current_frame_active());
    }

    #[test]
    fn shared_push_pop_records() {
        let s = SharedDepStack::new();
        s.push_frame();
        s.record_dep(NodeId(7));
        s.record_dep(NodeId(11));
        let frame = s.pop_frame();
        assert_eq!(frame, vec![NodeId(7), NodeId(11)]);
    }
}
