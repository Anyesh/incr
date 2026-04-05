//! Typed value arenas.
//!
//! Per section 5.2 of the concurrent core rewrite spec, node values live in
//! per-type arenas indexed by a type tag stored in `NodeData`. This file
//! implements the arena trait and two concrete implementations:
//!
//! - [`AtomicPrimitiveArena`] for `Copy` primitive types (u64, i64, f64, bool,
//!   and similar), which store values inline as atomic cells and support
//!   tear-free reads and writes without a state machine coordinating the
//!   value slot itself.
//! - [`GenericArena`] for everything else, where values are stored in
//!   `UnsafeCell<MaybeUninit<T>>` slots and access is gated by the node's
//!   state machine (section 7 of the spec). Readers only touch a slot when
//!   the corresponding node's state is `Clean`, which is guaranteed by the
//!   Release-Acquire pair on the state transition.
//!
//! Both arenas implement a common [`TypedArenaOps`] trait that exposes the
//! minimal operations the runtime needs: reserve a slot, write a value,
//! read a value, compare two slots for early cutoff equality. The runtime
//! stores arenas behind an [`ErasedArena`] trait object indexed by
//! `TypeId` in a `HashMap`.
//!
//! Status: stub. This file defines the trait signatures and placeholder
//! types. Implementations will be filled in when the state machine integrates
//! with node storage in the next session's work.

use std::any::TypeId;

/// Type-erased arena trait stored in the runtime's arena registry.
///
/// The runtime holds `Box<dyn ErasedArena>` keyed by `TypeId`, and downcasts
/// to the concrete arena type when it knows the value type statically (at
/// the `get::<T>` call site, which carries the type parameter).
#[allow(dead_code)]
pub(crate) trait ErasedArena: Send + Sync {
    /// Returns the `TypeId` of the value type this arena holds.
    fn erased_type_id(&self) -> TypeId;
}

// Placeholder. Real arena implementations land in the next session's work
// once the NodeData struct exists to reference them.
